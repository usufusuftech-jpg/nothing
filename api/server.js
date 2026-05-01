'use strict';

const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');

dotenv.config();

const app = express();

/* ============================================================
   MIDDLEWARE
============================================================ */
app.use(express.json({ limit: '16kb' }));

app.use(cors({
  origin: '*',
  methods: ['POST', 'GET', 'OPTIONS'],
  allowedHeaders: ['Content-Type'],
}));

/* ============================================================
   CONFIG
============================================================ */
const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY || '';

const GROQ_API_KEYS = [
  process.env.GROQ_API_KEY_1 || '',
  process.env.GROQ_API_KEY_2 || '',
  process.env.GROQ_API_KEY_3 || '',
].filter(Boolean);

const GROQ_MODEL = process.env.GROQ_MODEL || 'llama-3.3-70b-versatile';
const REQUEST_TIMEOUT_MS = 3000;

/* ============================================================
   SYSTEM PROMPT
============================================================ */
const SYSTEM_PROMPT = `
You are Nexus AI — the official intelligent assistant of InnoExpoGL.

You assist users with science, AI, coding, and research topics.
Be helpful, clear, and concise.
`;

/* ============================================================
   FETCH WITH TIMEOUT
============================================================ */
async function fetchWithTimeout(url, options, ms = REQUEST_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ms);

  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

/* ============================================================
   GEMINI
============================================================ */
async function tryGoogleGemini(userMsg, history) {
  if (!GOOGLE_API_KEY) throw new Error('No Gemini key');

  const historyText = history
    .map(m => `${m.role === 'user' ? 'User' : 'AI'}: ${m.content}`)
    .join('\n');

  const prompt = `${SYSTEM_PROMPT}\n\n${historyText}\nUser: ${userMsg}\nAI:`;

  const res = await fetchWithTimeout(
    `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${GOOGLE_API_KEY}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [{ parts: [{ text: prompt }] }],
      }),
    }
  );

  if (!res.ok) throw new Error(`Gemini ${res.status}`);

  const data = await res.json();
  return data?.candidates?.[0]?.content?.parts?.[0]?.text?.trim();
}

/* ============================================================
   GROQ
============================================================ */
async function tryGroq(apiKey, userMsg, history) {
  const res = await fetchWithTimeout(
    'https://api.groq.com/openai/v1/chat/completions',
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: GROQ_MODEL,
        messages: [
          { role: 'system', content: SYSTEM_PROMPT },
          ...history,
          { role: 'user', content: userMsg },
        ],
      }),
    }
  );

  if (!res.ok) throw new Error(`Groq ${res.status}`);

  const data = await res.json();
  return data?.choices?.[0]?.message?.content?.trim();
}

/* ============================================================
   AI ROUTER
============================================================ */
async function getAIResponse(message, history) {
  // Try Gemini first
  try {
    const reply = await tryGoogleGemini(message, history);
    if (reply) return { reply, provider: 'gemini' };
  } catch (_) {}

  // Fallback Groq keys
  for (let i = 0; i < GROQ_API_KEYS.length; i++) {
    try {
      const reply = await tryGroq(GROQ_API_KEYS[i], message, history);
      if (reply) return { reply, provider: `groq-${i + 1}` };
    } catch (_) {}
  }

  throw new Error('All providers failed');
}

/* ============================================================
   ROUTES (IMPORTANT)
   Vercel strips /api → so Express sees /chat
============================================================ */

// POST /api/chat
app.post('/chat', async (req, res) => {
  const { message, history } = req.body;

  if (!message) {
    return res.status(400).json({ error: 'message required' });
  }

  const safeHistory = Array.isArray(history) ? history.slice(-20) : [];

  try {
    const result = await getAIResponse(message, safeHistory);
    res.json(result);
  } catch (err) {
    res.status(503).json({
      error: 'AI service unavailable',
    });
  }
});

// GET /api/health
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    gemini: !!GOOGLE_API_KEY,
    groqKeys: GROQ_API_KEYS.length,
  });
});

/* ============================================================
   EXPORT FOR VERCEL
============================================================ */
module.exports = app;