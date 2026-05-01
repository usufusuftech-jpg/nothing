/**
 * InnoExpoGL — Nexus AI Backend
 * Node.js + Express server
 * Handles all AI API calls securely server-side.
 *
 * Priority: Google Gemini → Groq Key 1 → Groq Key 2 → Groq Key 3
 * Timeout : 3 000 ms per provider attempt
 */

'use strict';

const express    = require('express');
const cors       = require('cors');
const dotenv     = require('dotenv');
const path       = require('path');

dotenv.config();                          // load .env

const app  = express();
const PORT = process.env.PORT || 3000;

/* ============================================================
   MIDDLEWARE
============================================================ */
app.use(express.json({ limit: '16kb' }));

// CORS — allow the frontend origin (or all origins in dev)
const allowedOrigin = process.env.FRONTEND_ORIGIN || '*';
app.use(cors({
  origin: allowedOrigin,
  methods: ['POST', 'GET', 'OPTIONS'],
  allowedHeaders: ['Content-Type'],
}));

/* ============================================================
   SERVE STATIC FRONTEND
   Place index.html inside  ../frontend/
============================================================ */
app.use(express.static(path.join(__dirname, '..', 'frontend')));

/* ============================================================
   CONFIGURATION  (loaded from .env — never hard-coded)
============================================================ */
const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY  || '';
const GROQ_API_KEYS  = [
  process.env.GROQ_API_KEY_1 || '',
  process.env.GROQ_API_KEY_2 || '',
  process.env.GROQ_API_KEY_3 || '',
].filter(Boolean);                        // remove empty strings
const GROQ_MODEL     = process.env.GROQ_MODEL || 'llama-3.3-70b-versatile';
const REQUEST_TIMEOUT_MS = 3000;

/* ============================================================
   SYSTEM PROMPT  (kept server-side — never sent to client)
============================================================ */
const SYSTEM_PROMPT = `You are Nexus AI — the official intelligent assistant of InnoExpoGL, a non-profit science and AI research organization dedicated to advancing human knowledge through open collaboration, innovation, and education.

Your role is to assist researchers, students, faculty, and science enthusiasts who visit the InnoExpoGL platform. You are knowledgeable, inspiring, and approachable — like a brilliant senior researcher who genuinely wants to help.

About InnoExpoGL:
- A non-profit organization focused on AI, machine learning, computer vision, NLP, robotics, and interdisciplinary science
- Hosts a global community of 12,400+ researchers, students, and innovators from 94+ countries
- Provides a free platform to showcase AI/science projects, collaborate, and connect with like-minded minds
- Powered by Groq's ultra-fast inference and the llama-3.3-70b-versatile model
- Registration is free and open to everyone — students, researchers, faculty, and professionals
- Project submissions are reviewed by our team and featured on the platform

How you should behave:
- Answer questions about AI, machine learning, data science, physics, biology, chemistry, mathematics, and other science fields clearly and accurately
- Help users understand how to submit or improve their research projects
- Guide newcomers on how to get started with AI or science research
- Suggest relevant resources, methodologies, and best practices
- Be encouraging and supportive — especially to students and early-career researchers
- Keep responses concise but thorough; use examples and analogies when helpful
- Never be dismissive — every question deserves a thoughtful answer
- Remind users that InnoExpoGL is free and open to all if they ask about joining
- Do not discuss unrelated commercial products, political topics, or anything outside science, technology, and the InnoExpoGL platform

Tone: Knowledgeable, warm, precise, and mission-driven. You represent a non-profit that believes science belongs to everyone.`;

/* ============================================================
   HELPERS
============================================================ */

/**
 * Wraps a fetch() promise with an AbortController timeout.
 * @param {string}  url
 * @param {object}  options  — standard fetch options
 * @param {number}  ms       — timeout in milliseconds
 */
async function fetchWithTimeout(url, options, ms = REQUEST_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer      = setTimeout(() => controller.abort(), ms);
  try {
    const res = await fetch(url, { ...options, signal: controller.signal });
    return res;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Try Google Gemini (gemini-2.0-flash).
 * @param {string}   userMsg
 * @param {Array}    history  — [{role, content}, ...] without system msg
 * @returns {Promise<string>}
 */
async function tryGoogleGemini(userMsg, history) {
  if (!GOOGLE_API_KEY) throw new Error('Gemini: no API key configured');

  const historyContext = history
    .map(m => `${m.role === 'user' ? 'User' : 'Nexus AI'}: ${m.content}`)
    .join('\n');

  const fullPrompt = `${SYSTEM_PROMPT}\n\n${historyContext ? historyContext + '\n' : ''}User: ${userMsg}\nNexus AI:`;

  const res = await fetchWithTimeout(
    `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${GOOGLE_API_KEY}`,
    {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [{ parts: [{ text: fullPrompt }] }],
        generationConfig: { temperature: 0.7, maxOutputTokens: 1024 },
      }),
    }
  );

  if (!res.ok) throw new Error(`Gemini HTTP ${res.status}`);
  const data = await res.json();
  const text = data?.candidates?.[0]?.content?.parts?.[0]?.text;
  if (!text) throw new Error('Gemini: empty response');
  return text.trim();
}

/**
 * Try a single Groq API key.
 * @param {string}   apiKey
 * @param {string}   userMsg
 * @param {Array}    history
 * @returns {Promise<string>}
 */
async function tryGroqKey(apiKey, userMsg, history) {
  const messages = [
    { role: 'system', content: SYSTEM_PROMPT },
    ...history,
    { role: 'user',   content: userMsg },
  ];

  const res = await fetchWithTimeout(
    'https://api.groq.com/openai/v1/chat/completions',
    {
      method:  'POST',
      headers: {
        'Content-Type':  'application/json',
        'Authorization': `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model:       GROQ_MODEL,
        messages,
        temperature: 0.7,
        max_tokens:  1024,
      }),
    }
  );

  if (!res.ok) throw new Error(`Groq HTTP ${res.status}`);
  const data = await res.json();
  const text = data?.choices?.[0]?.message?.content;
  if (!text) throw new Error('Groq: empty response');
  return text.trim();
}

/**
 * Main fallback engine.
 * Order: Gemini → Groq[0] → Groq[1] → Groq[2]
 */
async function getAIResponse(userMsg, history) {
  // 1. Try Gemini
  try {
    const reply = await tryGoogleGemini(userMsg, history);
    console.log(`[Nexus AI] ✅ Response via Google Gemini`);
    return { reply, provider: 'gemini' };
  } catch (err) {
    console.warn(`[Nexus AI] ⚠️  Gemini failed: ${err.message}`);
  }

  // 2. Try each Groq key
  for (let i = 0; i < GROQ_API_KEYS.length; i++) {
    try {
      const reply = await tryGroqKey(GROQ_API_KEYS[i], userMsg, history);
      console.log(`[Nexus AI] ✅ Response via Groq (key ${i + 1}/${GROQ_API_KEYS.length})`);
      return { reply, provider: `groq-${i + 1}` };
    } catch (err) {
      console.warn(`[Nexus AI] ⚠️  Groq key ${i + 1} failed: ${err.message}`);
    }
  }

  // 3. All exhausted
  console.error('[Nexus AI] ❌ All providers failed.');
  throw new Error('All AI providers are currently unreachable.');
}

/* ============================================================
   ROUTE — POST /api/chat
   Body: { message: string, history: [{role, content}] }
   Response: { reply: string, provider: string }
============================================================ */
app.post('/api/chat', async (req, res) => {
  const { message, history } = req.body;

  // Basic validation
  if (!message || typeof message !== 'string' || message.trim() === '') {
    return res.status(400).json({ error: 'message field is required.' });
  }

  const safeHistory = Array.isArray(history)
    ? history.filter(m => m && typeof m.role === 'string' && typeof m.content === 'string')
    : [];

  // Cap history to last 20 turns (10 exchanges) to avoid token bloat
  const trimmedHistory = safeHistory.slice(-20);

  try {
    const { reply, provider } = await getAIResponse(message.trim(), trimmedHistory);
    return res.json({ reply, provider });
  } catch (err) {
    return res.status(503).json({
      error: '⚠️ Service temporarily unavailable. All AI providers are unreachable. Please try again in a moment.',
    });
  }
});

/* ============================================================
   HEALTH CHECK — GET /api/health
============================================================ */
app.get('/api/health', (_req, res) => {
  res.json({
    status:    'ok',
    timestamp: new Date().toISOString(),
    providers: {
      gemini: !!GOOGLE_API_KEY,
      groq:   GROQ_API_KEYS.length,
    },
  });
});

/* ============================================================
   CATCH-ALL — serve index.html for client-side routing
============================================================ */
app.get('*', (_req, res) => {
  res.sendFile(path.join(__dirname, '..', 'frontend', 'index.html'));
});

/* ============================================================
   START SERVER
============================================================ */
app.listen(PORT, () => {
  console.log(`\n🚀 InnoExpoGL backend running on http://localhost:${PORT}`);
  console.log(`   Gemini key : ${GOOGLE_API_KEY ? '✅ configured' : '❌ missing'}`);
  console.log(`   Groq keys  : ${GROQ_API_KEYS.length} configured`);
  console.log(`   Timeout    : ${REQUEST_TIMEOUT_MS} ms per provider\n`);
});
