import streamlit as st
import streamlit.components.v1 as components

# Set page configuration
st.set_page_config(page_title="Billo Mausi Birthday! 🐱", layout="centered")

# Copy the entire content of Vini.html into this variable
vini_html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Happy Birthday Billo Mausi! 🐱</title>
<link href="https://fonts.googleapis.com/css2?family=Bangers&family=Comic+Neue:ital,wght@0,400;0,700;1,700&display=swap" rel="stylesheet">
<style>
:root {
  --wa-green: #25D366;
  --wa-dark: #075E54;
  --wa-lite: #DCF8C6;
  --wa-bg: #E5DDD5;
  --vini: #FF7F00;
  --san: #9B59B6;
  --san-lt: #D7BDE2;
}
* { margin:0; padding:0; box-sizing:border-box; }
body {
  background-color: var(--wa-bg);
  background-image: repeating-linear-gradient(
    45deg,
    rgba(0,0,0,0.03) 0px, rgba(0,0,0,0.03) 1px,
    transparent 1px, transparent 12px
  ), repeating-linear-gradient(
    -45deg,
    rgba(0,0,0,0.03) 0px, rgba(0,0,0,0.03) 1px,
    transparent 1px, transparent 12px
  );
  font-family: 'Comic Neue', cursive;
  min-height: 100vh;
  padding-bottom: 40px;
}
 
/* ---- HEADER ---- */
.wa-header {
  background: linear-gradient(135deg, var(--wa-dark), #128C7E);
  padding: 14px 20px;
  display: flex; align-items: center; gap: 14px;
  position: sticky; top: 0; z-index: 100;
  box-shadow: 0 3px 12px rgba(0,0,0,0.3);
}
.wa-avatar {
  width: 46px; height: 46px; border-radius: 50%;
  background: linear-gradient(135deg, var(--vini), var(--san));
  display: flex; align-items: center; justify-content: center;
  font-size: 24px; border: 2px solid rgba(255,255,255,0.4);
  flex-shrink: 0;
}
.wa-header-info { flex: 1; }
.wa-header-name {
  font-family: 'Bangers', cursive; font-size: 20px; color: #fff;
  letter-spacing: 1px; line-height: 1;
}
.wa-header-sub { font-size: 11px; color: rgba(255,255,255,0.75); margin-top: 2px; }
.wa-header-icons { color: rgba(255,255,255,0.8); font-size: 20px; display: flex; gap: 16px; }
 
/* ---- TITLE CARD ---- */
.title-card {
  margin: 20px 16px 10px;
  background: linear-gradient(135deg, #1a1a1a, #2d1b69);
  border-radius: 18px;
  padding: 28px 20px;
  text-align: center;
  border: 3px solid #333;
  position: relative; overflow: hidden;
}
.title-card::before {
  content: '';
  position: absolute; inset: 0;
  background: repeating-linear-gradient(0deg, transparent, transparent 18px, rgba(255,255,255,0.03) 18px, rgba(255,255,255,0.03) 19px);
}
.title-main {
  font-family: 'Bangers', cursive;
  font-size: clamp(32px, 8vw, 56px);
  letter-spacing: 3px;
  color: #FFE566;
  text-shadow: 4px 4px 0 #FF4500, 7px 7px 0 rgba(0,0,0,0.4);
  line-height: 1;
  position: relative; z-index: 1;
}
.title-sub {
  font-family: 'Bangers', cursive;
  font-size: clamp(16px, 4vw, 24px);
  color: #ff7f00;
  letter-spacing: 2px;
  margin-top: 8px;
  position: relative; z-index: 1;
}
.title-label {
  display: inline-block;
  background: #FF4500;
  color: #fff;
  font-family: 'Bangers'; font-size: 13px; letter-spacing: 1px;
  padding: 3px 12px; border-radius: 30px;
  margin-top: 10px; position: relative; z-index: 1;
}
 
/* ---- DATE SEPARATOR ---- */
.date-sep {
  display: flex; align-items: center; justify-content: center;
  margin: 14px 16px;
}
.date-pill {
  background: rgba(0,0,0,0.18);
  color: rgba(0,0,0,0.6);
  font-size: 11px; font-weight: 700;
  padding: 4px 14px; border-radius: 20px;
}
 
/* ---- CHAT AREA ---- */
.chat { padding: 6px 14px; display: flex; flex-direction: column; gap: 10px; }
 
/* ---- MESSAGE WRAPPER ---- */
.msg { display: flex; align-items: flex-end; gap: 8px; max-width: 100%; }
.msg.vini { flex-direction: row-reverse; }
.msg.san { flex-direction: row; }
 
.avatar-sm {
  width: 32px; height: 32px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 16px; flex-shrink: 0; border: 2px solid;
}
.vini .avatar-sm { background: #FFF3E0; border-color: var(--vini); }
.san .avatar-sm  { background: #F3E5F5; border-color: var(--san); }
 
.bubble-wrap { max-width: min(78%, 400px); }
.sender-name {
  font-size: 11px; font-weight: 700; margin-bottom: 3px; padding-left: 12px;
}
.vini .sender-name { color: var(--vini); text-align: right; padding-right: 12px; padding-left: 0; }
.san  .sender-name { color: var(--san); }
 
.bubble {
  border-radius: 18px; padding: 10px 12px;
  position: relative; box-shadow: 0 2px 6px rgba(0,0,0,0.12);
}
.vini .bubble {
  background: var(--wa-lite);
  border-bottom-right-radius: 4px;
}
.san .bubble {
  background: #fff;
  border-bottom-left-radius: 4px;
}
.bubble-text {
  font-size: 13.5px; line-height: 1.5; color: #1a1a1a;
}
.bubble-time {
  font-size: 10px; color: #888; text-align: right; margin-top: 4px;
}
.vini .bubble-time { color: #4a8a60; }
 
/* ---- GIF PANEL ---- */
.gif-panel {
  border-radius: 14px; overflow: hidden;
  border: 2.5px solid rgba(0,0,0,0.15);
  background: #1a1a1a;
  margin-bottom: 4px;
  position: relative;
}
.gif-badge {
  position: absolute; top: 6px; left: 8px;
  background: rgba(0,0,0,0.6); color: #fff;
  font-size: 9px; font-weight: 700; letter-spacing: 1px;
  padding: 2px 7px; border-radius: 4px; z-index: 10;
}
.gif-canvas {
  width: 100%; height: 160px;
  display: flex; align-items: center; justify-content: center;
  position: relative; overflow: hidden;
}
.gif-caption {
  background: rgba(0,0,0,0.7); color: #fff;
  font-size: 11px; font-weight: 700; text-align: center;
  padding: 5px 10px;
  font-family: 'Comic Neue', cursive;
}
 
/* ---- REACTION TEXT ---- */
.reaction {
  font-size: 22px; padding: 4px 8px; text-align: center;
  display: block;
}
 
/* ---- CAT SVG ANIMATIONS ---- */
 
/* Vini punch */
@keyframes vini-arm-punch {
  0%,100% { transform: rotate(-10deg) translateX(0); }
  30% { transform: rotate(45deg) translateX(18px); }
  50% { transform: rotate(40deg) translateX(16px); }
  70% { transform: rotate(-5deg) translateX(0); }
}
@keyframes pow-pop {
  0%,100% { transform: scale(0); opacity: 0; }
  35%,55% { transform: scale(1.1); opacity: 1; }
  65% { transform: scale(0.9); opacity: 0.7; }
}
@keyframes cat-shimmy { 0%,100%{transform:translateX(0)rotate(0)} 20%{transform:translateX(-4px)rotate(-3deg)} 60%{transform:translateX(4px)rotate(3deg)} }
@keyframes body-bounce { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-5px)} }
 
/* San fall */
@keyframes san-fall {
  0% { transform: translateY(-80px) rotate(0deg); opacity: 1; }
  60% { transform: translateY(10px) rotate(380deg); opacity: 1; }
  70%,90% { transform: translateY(15px) rotate(360deg) scaleX(1.4) scaleY(0.4); opacity: 1; }
  95%,100% { transform: translateY(-80px) rotate(0deg); opacity: 0; }
}
@keyframes dust-puff {
  0%,60% { opacity: 0; transform: scale(0); }
  70% { opacity: 0.8; transform: scale(1); }
  95% { opacity: 0; transform: scale(1.5); }
  100% { opacity: 0; }
}
@keyframes star-spin {
  0% { transform: rotate(0deg) translateX(20px); opacity: 1; }
  100% { transform: rotate(360deg) translateX(20px); opacity: 0; }
}
 
/* Weird smile */
@keyframes creep-smile {
  0%,100% { d: path("M 32 70 Q 50 74 68 70"); }
  40%,60% { d: path("M 22 65 Q 50 85 78 65"); }
}
@keyframes eye-twitch {
  0%,100%,20%,40% { transform: scaleX(1) scaleY(1); }
  10% { transform: scaleX(1.3) scaleY(0.4); }
  30% { transform: scaleX(0.8) scaleY(1.2); }
}
@keyframes head-tilt {
  0%,100% { transform: rotate(0deg); }
  33% { transform: rotate(-8deg); }
  66% { transform: rotate(10deg); }
}
 
/* Laugh bounce */
@keyframes laugh-bounce {
  0%,100% { transform: translateY(0) rotate(0); }
  15% { transform: translateY(-18px) rotate(-4deg); }
  30% { transform: translateY(-12px) rotate(3deg); }
  45% { transform: translateY(-20px) rotate(-3deg); }
  60% { transform: translateY(-8px) rotate(4deg); }
  75% { transform: translateY(-16px) rotate(-2deg); }
}
@keyframes tears { 0%,100%{opacity:0;transform:translateY(-5px)} 30%,70%{opacity:1;transform:translateY(10px)} }
@keyframes ha-pop {
  0%,100% { opacity: 0; transform: scale(0) translateY(0); }
  20%,40% { opacity: 1; transform: scale(1) translateY(-12px); }
  50% { opacity: 0; transform: scale(0.5) translateY(-20px); }
}
 
/* Fight cloud */
@keyframes fight-shake {
  0%,100% { transform: translate(0,0) rotate(0deg); }
  10% { transform: translate(-6px,3px) rotate(-5deg); }
  20% { transform: translate(5px,-4px) rotate(4deg); }
  30% { transform: translate(-4px,5px) rotate(-3deg); }
  40% { transform: translate(6px,-3px) rotate(6deg); }
  50% { transform: translate(-5px,4px) rotate(-4deg); }
  60% { transform: translate(4px,-6px) rotate(3deg); }
  70% { transform: translate(-6px,3px) rotate(-5deg); }
  80% { transform: translate(5px,5px) rotate(4deg); }
  90% { transform: translate(-3px,-4px) rotate(-2deg); }
}
@keyframes limb-flail { 0%,100%{transform:rotate(0deg)} 50%{transform:rotate(180deg)} }
@keyframes star-burst { 0%{transform:scale(0)rotate(0);opacity:1} 100%{transform:scale(1.5)rotate(90deg);opacity:0} }
 
/* Happy dance */
@keyframes happy-sway {
  0%,100% { transform: rotate(-6deg) translateX(-3px); }
  50% { transform: rotate(6deg) translateX(3px); }
}
@keyframes confetti-fall {
  0% { transform: translateY(-30px) rotate(0deg); opacity: 1; }
  100% { transform: translateY(180px) rotate(720deg); opacity: 0; }
}
@keyframes banner-wave {
  0%,100% { transform: skewX(-2deg); }
  50% { transform: skewX(2deg); }
}
@keyframes heart-float {
  0% { transform: translateY(0) scale(1); opacity: 1; }
  100% { transform: translateY(-60px) scale(0.5); opacity: 0; }
}
@keyframes glow-pulse { 0%,100%{opacity:0.6} 50%{opacity:1} }
 
/* ---- BIG BDAY SECTION ---- */
.bday-finale {
  margin: 16px;
  background: linear-gradient(135deg, #FF006E, #FB5607 40%, #FFBE0B);
  border-radius: 22px;
  padding: 30px 20px;
  text-align: center;
  border: 4px solid #1a1a1a;
  box-shadow: 6px 6px 0 rgba(0,0,0,0.3);
  position: relative; overflow: hidden;
}
.bday-finale::before {
  content: '';
  position: absolute; inset: 0;
  background: repeating-linear-gradient(45deg, transparent, transparent 10px, rgba(255,255,255,0.05) 10px, rgba(255,255,255,0.05) 11px);
}
.bday-title {
  font-family: 'Bangers', cursive;
  font-size: clamp(28px, 7vw, 52px);
  color: #fff;
  text-shadow: 3px 3px 0 rgba(0,0,0,0.4);
  letter-spacing: 3px;
  line-height: 1.1;
  position: relative; z-index: 1;
}
.bday-subtitle {
  font-family: 'Bangers', cursive;
  font-size: clamp(16px, 4vw, 26px);
  color: #1a1a1a;
  letter-spacing: 2px;
  margin-top: 6px;
  position: relative; z-index: 1;
}
.bday-msg {
  background: rgba(255,255,255,0.22);
  border-radius: 14px;
  padding: 14px 18px;
  margin-top: 16px;
  font-size: clamp(14px, 3.5vw, 18px);
  color: #fff;
  font-weight: 700;
  line-height: 1.6;
  border: 2px solid rgba(255,255,255,0.3);
  position: relative; z-index: 1;
  text-shadow: 1px 1px 0 rgba(0,0,0,0.3);
}
.from-tag {
  display: inline-block;
  background: #1a1a1a;
  color: #FFE566;
  font-family: 'Bangers'; font-size: 15px; letter-spacing: 2px;
  padding: 5px 18px; border-radius: 30px;
  margin-top: 14px; position: relative; z-index: 1;
}
.friendship-badge {
  display: flex; align-items: center; justify-content: center;
  gap: 8px; margin: 16px 0 0;
  font-family: 'Bangers'; font-size: 18px; letter-spacing: 1px;
  color: #1a1a1a; position: relative; z-index: 1;
}
</style>
</head>
<body>
 
<!-- WhatsApp header -->
<div class="wa-header">
  <div class="wa-avatar">🐱</div>
  <div class="wa-header-info">
    <div class="wa-header-name">San & Vini: WAR ZONE 🥊</div>
    <div class="wa-header-sub">2 members · Billo Mausi Birthday Edition</div>
  </div>
  <div class="wa-header-icons">📹 📞</div>
</div>
 
<!-- Title card -->
<div class="title-card">
  <div class="title-main">SAN vs VINI</div>
  <div class="title-sub">THE ETERNAL WHATSAPP WAR 🐱⚡🐱</div>
  <div class="title-label">TOM & JERRY CHRONICLES — BIRTHDAY SPECIAL</div>
</div>
 
<div class="date-sep"><div class="date-pill">TODAY · BIRTHDAY OF BILLO MAUSI 🎂</div></div>
 
<!-- ===== CHAT ===== -->
<div class="chat">
 
  <!-- 1. Vini opens with a punch GIF -->
  <div class="msg vini">
    <div class="avatar-sm">🦊</div>
    <div class="bubble-wrap">
      <div class="sender-name">Vini</div>
      <div class="bubble">
        <div class="gif-panel">
          <div class="gif-badge">GIF</div>
          <div class="gif-canvas" style="background:linear-gradient(135deg,#FF4500,#FF7F00)">
            <!-- Vini cat punching -->
            <svg width="220" height="155" viewBox="0 0 220 155" style="overflow:visible">
              <!-- Speed lines -->
              <g opacity="0.3" stroke="#FFE566" stroke-width="1.5">
                <line x1="140" y1="75" x2="210" y2="65"/><line x1="140" y1="80" x2="215" y2="80"/>
                <line x1="140" y1="85" x2="210" y2="95"/><line x1="138" y1="70" x2="205" y2="55"/>
              </g>
              <!-- Body group with shimmy -->
              <g style="animation:cat-shimmy 1.2s infinite; transform-origin:85px 85px">
                <!-- Cat body (orange) -->
                <ellipse cx="85" cy="100" rx="28" ry="22" fill="#FF7F00"/>
                <!-- Tail -->
                <path d="M 57 110 Q 30 130 40 148 Q 50 155 55 148 Q 48 138 65 120" fill="none" stroke="#FF6500" stroke-width="8" stroke-linecap="round"/>
                <!-- Legs -->
                <rect x="68" y="118" width="10" height="18" rx="5" fill="#FF6500"/>
                <rect x="92" y="118" width="10" height="18" rx="5" fill="#FF6500"/>
                <!-- Head -->
                <circle cx="85" cy="62" r="30" fill="#FF7F00"/>
                <!-- Ears -->
                <polygon points="58,46 65,22 78,46" fill="#FF7F00"/>
                <polygon points="92,46 105,22 112,46" fill="#FF7F00"/>
                <polygon points="61,45 65,28 76,45" fill="#FFB3B3" opacity="0.7"/>
                <polygon points="94,45 105,28 110,45" fill="#FFB3B3" opacity="0.7"/>
                <!-- Eyes (smug) -->
                <ellipse cx="73" cy="62" rx="8" ry="7" fill="#222"/>
                <ellipse cx="97" cy="62" rx="8" ry="7" fill="#222"/>
                <circle cx="75" cy="60" r="2.5" fill="white"/>
                <circle cx="99" cy="60" r="2.5" fill="white"/>
                <!-- Smug half-closed eyes -->
                <rect x="65" y="57" width="16" height="5" rx="2" fill="#FF7F00" opacity="0.85"/>
                <rect x="89" y="57" width="16" height="5" rx="2" fill="#FF7F00" opacity="0.85"/>
                <!-- Nose -->
                <ellipse cx="85" cy="72" rx="4" ry="3" fill="#FF1493"/>
                <!-- Smirk -->
                <path d="M 78 79 Q 88 86 96 80" stroke="#333" fill="none" stroke-width="2" stroke-linecap="round"/>
                <!-- Whiskers -->
                <line x1="50" y1="70" x2="77" y2="73" stroke="#fff" stroke-width="1.2" opacity="0.8"/>
                <line x1="48" y1="76" x2="77" y2="76" stroke="#fff" stroke-width="1.2" opacity="0.8"/>
                <line x1="93" y1="73" x2="120" y2="70" stroke="#fff" stroke-width="1.2" opacity="0.8"/>
                <line x1="93" y1="76" x2="122" y2="76" stroke="#fff" stroke-width="1.2" opacity="0.8"/>
                <!-- Left arm -->
                <path d="M 60 90 Q 45 100 50 112" stroke="#FF6500" stroke-width="9" fill="none" stroke-linecap="round"/>
                <!-- Punching arm -->
                <g style="animation:vini-arm-punch 1.4s infinite; transform-origin:110px 90px">
                  <path d="M 110 90 Q 140 82 160 80" stroke="#FF6500" stroke-width="10" fill="none" stroke-linecap="round"/>
                  <!-- Fist -->
                  <ellipse cx="164" cy="79" rx="13" ry="11" fill="#FF7F00" stroke="#FF6500" stroke-width="2"/>
                  <line x1="158" y1="82" x2="170" y2="82" stroke="#FF6500" stroke-width="1.5"/>
                  <line x1="158" y1="78" x2="170" y2="78" stroke="#FF6500" stroke-width="1.5"/>
                </g>
              </g>
              <!-- POW! effect -->
              <g style="animation:pow-pop 1.4s infinite; transform-origin:190px 60px">
                <polygon points="190,38 197,55 215,48 200,62 215,75 197,68 190,85 183,68 165,75 180,62 165,48 183,55" fill="#FFE566" stroke="#FF4500" stroke-width="2"/>
                <text x="190" y="67" text-anchor="middle" font-family="Bangers,cursive" font-size="15" fill="#FF4500">POW!</text>
              </g>
            </svg>
          </div>
          <div class="gif-caption">🥊 Standard Monday greeting</div>
        </div>
        <div class="bubble-text">Good morning 😇</div>
        <div class="bubble-time">08:42 ✓✓</div>
      </div>
    </div>
  </div>
 
  <!-- 2. San falls dramatically -->
  <div class="msg san">
    <div class="avatar-sm">🐱</div>
    <div class="bubble-wrap">
      <div class="sender-name">San (Billo Mausi)</div>
      <div class="bubble">
        <div class="gif-panel">
          <div class="gif-badge">GIF</div>
          <div class="gif-canvas" style="background:linear-gradient(135deg,#4a148c,#7B1FA2)">
            <svg width="220" height="155" viewBox="0 0 220 155" style="overflow:visible">
              <!-- Floor -->
              <line x1="10" y1="148" x2="210" y2="148" stroke="#9C27B0" stroke-width="3"/>
              <!-- Dust -->
              <g style="animation:dust-puff 2s infinite">
                <ellipse cx="110" cy="145" rx="35" ry="8" fill="#CE93D8" opacity="0.5"/>
                <ellipse cx="95" cy="142" rx="14" ry="5" fill="#E1BEE7" opacity="0.4"/>
                <ellipse cx="128" cy="142" rx="12" ry="5" fill="#E1BEE7" opacity="0.4"/>
              </g>
              <!-- Stars -->
              <g style="animation:star-spin 2s infinite linear; transform-origin:110px 120px">
                <text x="80" y="115" font-size="14">⭐</text>
              </g>
              <g style="animation:star-spin 2s 0.5s infinite linear; transform-origin:110px 120px">
                <text x="120" y="120" font-size="12">✦</text>
              </g>
              <g style="animation:star-spin 2s 1s infinite linear; transform-origin:110px 120px">
                <text x="100" y="108" font-size="10">★</text>
              </g>
              <!-- Falling San cat -->
              <g style="animation:san-fall 2s infinite ease-in; transform-origin:110px 75px">
                <circle cx="110" cy="55" r="30" fill="#9B59B6"/>
                <polygon points="86,38 93,15 103,38" fill="#9B59B6"/>
                <polygon points="117,38 127,15 134,38" fill="#9B59B6"/>
                <polygon points="89,38 93,22 101,38" fill="#CE93D8" opacity="0.7"/>
                <polygon points="119,38 127,22 132,38" fill="#CE93D8" opacity="0.7"/>
                <!-- Dizzy X eyes -->
                <text x="96" y="58" font-size="14" fill="#fff" font-weight="bold">✕</text>
                <text x="112" y="58" font-size="14" fill="#fff" font-weight="bold">✕</text>
                <!-- Open O mouth -->
                <ellipse cx="110" cy="68" rx="7" ry="8" fill="#fff" opacity="0.9"/>
                <!-- Whiskers -->
                <line x1="78" y1="60" x2="100" y2="62" stroke="#fff" stroke-width="1.2" opacity="0.7"/>
                <line x1="78" y1="65" x2="100" y2="65" stroke="#fff" stroke-width="1.2" opacity="0.7"/>
                <line x1="120" y1="62" x2="142" y2="60" stroke="#fff" stroke-width="1.2" opacity="0.7"/>
                <line x1="120" y1="65" x2="142" y2="65" stroke="#fff" stroke-width="1.2" opacity="0.7"/>
                <!-- Body flailing -->
                <ellipse cx="110" cy="88" rx="20" ry="16" fill="#8E44AD"/>
                <!-- Legs up in air -->
                <path d="M 96 97 Q 85 115 78 108" stroke="#8E44AD" stroke-width="8" fill="none" stroke-linecap="round"/>
                <path d="M 124 97 Q 135 115 142 108" stroke="#8E44AD" stroke-width="8" fill="none" stroke-linecap="round"/>
                <!-- AAAA text -->
                <text x="110" y="30" text-anchor="middle" font-family="Bangers,cursive" font-size="13" fill="#FFE566" opacity="0.9">A A A A A!</text>
              </g>
            </svg>
          </div>
          <div class="gif-caption">🙀 Accurate representation of my dignity</div>
        </div>
        <div class="bubble-text">My reaction every time you text 😭</div>
        <div class="bubble-time">08:43 ✓✓</div>
      </div>
    </div>
  </div>
 
  <!-- 3. Vini - smug weird smile -->
  <div class="msg vini">
    <div class="avatar-sm">🦊</div>
    <div class="bubble-wrap">
      <div class="sender-name">Vini</div>
      <div class="bubble">
        <div class="gif-panel">
          <div class="gif-badge">GIF</div>
          <div class="gif-canvas" style="background:linear-gradient(135deg,#1B5E20,#2E7D32)">
            <svg width="220" height="155" viewBox="0 0 220 155" style="overflow:visible">
              <!-- Creepy spotlight -->
              <defs>
                <radialGradient id="spot" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stop-color="#FFE566" stop-opacity="0.3"/>
                  <stop offset="100%" stop-color="transparent"/>
                </radialGradient>
              </defs>
              <ellipse cx="110" cy="80" rx="80" ry="70" fill="url(#spot)"/>
              <!-- Head with tilt animation -->
              <g style="animation:head-tilt 1.8s infinite ease-in-out; transform-origin:110px 72px">
                <!-- Body -->
                <ellipse cx="110" cy="118" rx="26" ry="20" fill="#FF7F00"/>
                <!-- Arms on hips - smug -->
                <path d="M 86 110 Q 70 120 72 132" stroke="#FF6500" stroke-width="8" fill="none" stroke-linecap="round"/>
                <path d="M 134 110 Q 150 120 148 132" stroke="#FF6500" stroke-width="8" fill="none" stroke-linecap="round"/>
                <!-- Legs -->
                <rect x="96" y="134" width="10" height="15" rx="5" fill="#FF6500"/>
                <rect x="114" y="134" width="10" height="15" rx="5" fill="#FF6500"/>
                <!-- Head -->
                <circle cx="110" cy="72" r="32" fill="#FF7F00"/>
                <!-- Ears -->
                <polygon points="82,56 90,30 103,56" fill="#FF7F00"/>
                <polygon points="117,56 130,30 138,56" fill="#FF7F00"/>
                <polygon points="85,55 90,34 101,55" fill="#FFB3B3" opacity="0.6"/>
                <polygon points="119,55 130,34 136,55" fill="#FFB3B3" opacity="0.6"/>
                <!-- Twitchy left eye -->
                <g style="animation:eye-twitch 1.8s infinite; transform-origin:98px 70px">
                  <ellipse cx="98" cy="70" rx="9" ry="9" fill="#333"/>
                  <circle cx="100" cy="67" r="3" fill="white"/>
                  <circle cx="100" cy="67" r="1.5" fill="#111"/>
                </g>
                <!-- Creepy wide right eye -->
                <ellipse cx="122" cy="70" rx="11" ry="12" fill="#333"/>
                <circle cx="124" cy="66" r="4" fill="white"/>
                <circle cx="124" cy="66" r="2" fill="#111"/>
                <!-- Sparkle in eye -->
                <circle cx="126" cy="64" r="1.5" fill="white"/>
                <!-- Nose -->
                <ellipse cx="110" cy="80" rx="4" ry="3" fill="#FF1493"/>
                <!-- THE SMILE - grows creepily -->
                <path id="smilePath" d="M 86 88 Q 110 104 134 88" stroke="#1a1a1a" fill="none" stroke-width="3" stroke-linecap="round"
                  style="animation:creep-smile 1.8s infinite ease-in-out"/>
                <!-- Whiskers extra long -->
                <line x1="55" y1="76" x2="98" y2="80" stroke="#fff" stroke-width="1.5" opacity="0.7"/>
                <line x1="52" y1="83" x2="98" y2="83" stroke="#fff" stroke-width="1.5" opacity="0.7"/>
                <line x1="122" y1="80" x2="165" y2="76" stroke="#fff" stroke-width="1.5" opacity="0.7"/>
                <line x1="122" y1="83" x2="168" y2="83" stroke="#fff" stroke-width="1.5" opacity="0.7"/>
              </g>
              <!-- Heh heh text -->
              <text x="28" y="38" font-family="Bangers,cursive" font-size="16" fill="#FFE566" opacity="0.8">heh heh...</text>
              <text x="148" y="32" font-family="Bangers,cursive" font-size="13" fill="#a5d6a7" opacity="0.8">hehehe</text>
            </svg>
          </div>
          <div class="gif-caption">😈 When I win a vaad-vivaad</div>
        </div>
        <div class="bubble-text">Hehehehehe 😏</div>
        <div class="bubble-time">08:44 ✓✓</div>
      </div>
    </div>
  </div>
 
  <!-- 4. San laughing back -->
  <div class="msg san">
    <div class="avatar-sm">🐱</div>
    <div class="bubble-wrap">
      <div class="sender-name">San (Billo Mausi)</div>
      <div class="bubble">
        <div class="gif-panel">
          <div class="gif-badge">GIF</div>
          <div class="gif-canvas" style="background:linear-gradient(135deg,#0D47A1,#1976D2)">
            <svg width="220" height="155" viewBox="0 0 220 155" style="overflow:visible">
              <!-- Bounce-laughing San cat -->
              <g style="animation:laugh-bounce 0.7s infinite; transform-origin:110px 85px">
                <!-- Body -->
                <ellipse cx="110" cy="112" rx="28" ry="22" fill="#9B59B6"/>
                <!-- Arms holding sides -->
                <path d="M 84 105 Q 70 115 74 125" stroke="#8E44AD" stroke-width="9" fill="none" stroke-linecap="round"/>
                <path d="M 136 105 Q 150 115 146 125" stroke="#8E44AD" stroke-width="9" fill="none" stroke-linecap="round"/>
                <!-- Legs -->
                <rect x="96" y="130" width="11" height="16" rx="5" fill="#8E44AD"/>
                <rect x="113" y="130" width="11" height="16" rx="5" fill="#8E44AD"/>
                <!-- Head -->
                <circle cx="110" cy="68" r="30" fill="#9B59B6"/>
                <!-- Ears -->
                <polygon points="84,52 92,26 103,52" fill="#9B59B6"/>
                <polygon points="117,52 128,26 136,52" fill="#9B59B6"/>
                <polygon points="87,51 92,30 101,51" fill="#CE93D8" opacity="0.7"/>
                <polygon points="119,51 128,30 134,51" fill="#CE93D8" opacity="0.7"/>
                <!-- Squinting laugh eyes - crescents -->
                <path d="M 89 65 Q 98 60 107 65" stroke="#fff" fill="none" stroke-width="3" stroke-linecap="round"/>
                <path d="M 113 65 Q 122 60 131 65" stroke="#fff" fill="none" stroke-width="3" stroke-linecap="round"/>
                <!-- Huge open laugh mouth -->
                <ellipse cx="110" cy="80" rx="14" ry="10" fill="#fff" opacity="0.95"/>
                <ellipse cx="110" cy="84" rx="10" ry="6" fill="#FF69B4" opacity="0.8"/>
                <!-- Whiskers -->
                <line x1="72" y1="68" x2="97" y2="70" stroke="#fff" stroke-width="1.2" opacity="0.7"/>
                <line x1="70" y1="74" x2="97" y2="74" stroke="#fff" stroke-width="1.2" opacity="0.7"/>
                <line x1="123" y1="70" x2="148" y2="68" stroke="#fff" stroke-width="1.2" opacity="0.7"/>
                <line x1="123" y1="74" x2="150" y2="74" stroke="#fff" stroke-width="1.2" opacity="0.7"/>
                <!-- Laugh tears -->
                <g style="animation:tears 0.7s infinite">
                  <ellipse cx="90" cy="72" rx="3" ry="5" fill="#4FC3F7" opacity="0.8"/>
                  <ellipse cx="130" cy="72" rx="3" ry="5" fill="#4FC3F7" opacity="0.8"/>
                </g>
              </g>
              <!-- HA HA HA floating text -->
              <text x="24" y="32" font-family="Bangers,cursive" font-size="18" fill="#FFE566" style="animation:ha-pop 0.7s infinite">HA</text>
              <text x="80" y="22" font-family="Bangers,cursive" font-size="18" fill="#FFE566" style="animation:ha-pop 0.7s 0.23s infinite">HA</text>
              <text x="138" y="32" font-family="Bangers,cursive" font-size="18" fill="#FFE566" style="animation:ha-pop 0.7s 0.46s infinite">HA</text>
              <text x="174" y="22" font-family="Bangers,cursive" font-size="14" fill="#80DEEA" style="animation:ha-pop 0.7s 0.12s infinite">😂</text>
            </svg>
          </div>
          <div class="gif-caption">😹 Me after every vaad-vivaad I WIN</div>
        </div>
        <div class="bubble-text">Hahahaha okay okay 😂😂</div>
        <div class="bubble-time">08:45 ✓✓</div>
      </div>
    </div>
  </div>
 
  <!-- 5. The classic fight cloud -->
  <div class="msg vini">
    <div class="avatar-sm">🦊</div>
    <div class="bubble-wrap">
      <div class="sender-name">Vini</div>
      <div class="bubble">
        <div class="gif-panel">
          <div class="gif-badge">GIF</div>
          <div class="gif-canvas" style="background:linear-gradient(135deg,#B71C1C,#D32F2F)">
            <svg width="220" height="155" viewBox="0 0 220 155" style="overflow:visible">
              <!-- Fight cloud (shaking) -->
              <g style="animation:fight-shake 0.18s infinite; transform-origin:110px 75px">
                <!-- Cloud of dust -->
                <ellipse cx="110" cy="88" rx="65" ry="55" fill="#FF8A65" opacity="0.7"/>
                <ellipse cx="90" cy="75" rx="45" ry="38" fill="#FFAB91" opacity="0.6"/>
                <ellipse cx="130" cy="80" rx="42" ry="35" fill="#FF8A65" opacity="0.5"/>
                <ellipse cx="108" cy="92" rx="52" ry="44" fill="#FF7043" opacity="0.4"/>
                <!-- Orange cat limbs sticking out -->
                <!-- Fist left -->
                <ellipse cx="44" cy="68" rx="14" ry="12" fill="#FF7F00" stroke="#FF6500" stroke-width="2"/>
                <line x1="44" y1="72" x2="44" y2="65" stroke="#FF6500" stroke-width="1.5"/>
                <!-- Foot bottom -->
                <ellipse cx="100" cy="148" rx="16" ry="10" fill="#FF7F00" stroke="#FF6500" stroke-width="2"/>
                <!-- Ear top -->
                <polygon points="80,28 88,10 96,28" fill="#FF7F00"/>
                <!-- Purple cat limbs -->
                <!-- Fist right -->
                <ellipse cx="176" cy="72" rx="14" ry="12" fill="#9B59B6" stroke="#8E44AD" stroke-width="2"/>
                <line x1="176" y1="76" x2="176" y2="69" stroke="#8E44AD" stroke-width="1.5"/>
                <!-- Foot bottom right -->
                <ellipse cx="130" cy="148" rx="16" ry="10" fill="#9B59B6" stroke="#8E44AD" stroke-width="2"/>
                <!-- Ear top right -->
                <polygon points="126,28 134,10 142,28" fill="#9B59B6"/>
                <!-- Tail orange -->
                <path d="M 52 100 Q 30 130 40 148" fill="none" stroke="#FF6500" stroke-width="8" stroke-linecap="round"
                  style="animation:limb-flail 0.4s infinite; transform-origin:52px 100px"/>
                <!-- Tail purple -->
                <path d="M 168 100 Q 190 130 180 148" fill="none" stroke="#8E44AD" stroke-width="8" stroke-linecap="round"
                  style="animation:limb-flail 0.4s 0.2s infinite; transform-origin:168px 100px"/>
              </g>
              <!-- Action stars burst -->
              <g style="animation:star-burst 0.6s infinite; transform-origin:110px 60px">
                <polygon points="110,36 115,52 132,52 118,62 124,78 110,68 96,78 102,62 88,52 105,52" fill="#FFE566" opacity="0.9"/>
              </g>
              <g style="animation:star-burst 0.6s 0.3s infinite; transform-origin:55px 45px">
                <polygon points="55,28 58,38 68,38 60,44 63,54 55,48 47,54 50,44 42,38 52,38" fill="#FFE566" opacity="0.8"/>
              </g>
              <!-- SMACK text -->
              <text x="150" y="42" font-family="Bangers,cursive" font-size="20" fill="#FFE566" 
                transform="rotate(-15,150,42)" style="animation:pow-pop 0.6s infinite">SMACK!</text>
              <text x="22" y="50" font-family="Bangers,cursive" font-size="16" fill="#80CBC4"
                transform="rotate(10,22,50)" style="animation:pow-pop 0.6s 0.3s infinite">BONK!</text>
            </svg>
          </div>
          <div class="gif-caption">💥 Us, every. single. day.</div>
        </div>
        <div class="bubble-text">Standard friendship activities 🐱🥊🐱</div>
        <div class="bubble-time">08:46 ✓✓</div>
      </div>
    </div>
  </div>
 
  <!-- 6. San sends weird falling GIF back -->
  <div class="msg san">
    <div class="avatar-sm">🐱</div>
    <div class="bubble-wrap">
      <div class="sender-name">San (Billo Mausi)</div>
      <div class="bubble">
        <div class="gif-panel">
          <div class="gif-badge">GIF</div>
          <div class="gif-canvas" style="background:linear-gradient(135deg,#004D40,#00695C)">
            <svg width="220" height="155" viewBox="0 0 220 155" style="overflow:visible">
              <!-- San cat strutting proudly -->
              <g style="animation:happy-sway 0.8s infinite ease-in-out; transform-origin:110px 80px">
                <!-- Body -->
                <ellipse cx="110" cy="112" rx="27" ry="21" fill="#9B59B6"/>
                <!-- Legs walking -->
                <rect x="96" y="128" width="10" height="18" rx="5" fill="#8E44AD" transform="rotate(-10,101,137)"/>
                <rect x="114" y="128" width="10" height="18" rx="5" fill="#8E44AD" transform="rotate(10,119,137)"/>
                <!-- Arm raised with trophy -->
                <path d="M 135 104 Q 155 90 158 76" stroke="#8E44AD" stroke-width="8" fill="none" stroke-linecap="round"/>
                <!-- Trophy 🏆 -->
                <text x="150" y="70" font-size="22">🏆</text>
                <!-- Other arm flapping -->
                <path d="M 85 104 Q 65 94 62 105" stroke="#8E44AD" stroke-width="8" fill="none" stroke-linecap="round"
                  style="animation:limb-flail 0.8s infinite; transform-origin:85px 104px"/>
                <!-- Head -->
                <circle cx="110" cy="68" r="30" fill="#9B59B6"/>
                <!-- Ears -->
                <polygon points="84,52 92,28 103,52" fill="#9B59B6"/>
                <polygon points="117,52 128,28 136,52" fill="#9B59B6"/>
                <polygon points="87,51 92,32 101,51" fill="#CE93D8" opacity="0.7"/>
                <polygon points="119,51 128,32 134,51" fill="#CE93D8" opacity="0.7"/>
                <!-- Happy sparkle eyes -->
                <ellipse cx="97" cy="67" rx="8" ry="8" fill="#fff"/>
                <ellipse cx="123" cy="67" rx="8" ry="8" fill="#fff"/>
                <ellipse cx="97" cy="67" rx="5" ry="5" fill="#1a1a1a"/>
                <ellipse cx="123" cy="67" rx="5" ry="5" fill="#1a1a1a"/>
                <!-- Stars in eyes -->
                <text x="91" y="71" font-size="8" fill="#FFE566">★</text>
                <text x="117" y="71" font-size="8" fill="#FFE566">★</text>
                <!-- Big smile -->
                <path d="M 90 80 Q 110 96 130 80" stroke="#fff" fill="none" stroke-width="3" stroke-linecap="round"/>
                <!-- Nose -->
                <ellipse cx="110" cy="74" rx="4" ry="3" fill="#FF69B4"/>
                <!-- Whiskers -->
                <line x1="72" y1="68" x2="97" y2="70" stroke="#fff" stroke-width="1.2" opacity="0.7"/>
                <line x1="70" y1="74" x2="97" y2="74" stroke="#fff" stroke-width="1.2" opacity="0.7"/>
                <line x1="123" y1="70" x2="148" y2="68" stroke="#fff" stroke-width="1.2" opacity="0.7"/>
                <line x1="123" y1="74" x2="150" y2="74" stroke="#fff" stroke-width="1.2" opacity="0.7"/>
              </g>
              <!-- Winner sparkles -->
              <text x="20" y="40" font-size="14" style="animation:ha-pop 0.8s infinite">⭐</text>
              <text x="175" y="35" font-size="16" style="animation:ha-pop 0.8s 0.4s infinite">✨</text>
              <text x="35" y="130" font-size="12" style="animation:ha-pop 0.8s 0.2s infinite">💫</text>
              <text x="168" y="120" font-size="14" style="animation:ha-pop 0.8s 0.6s infinite">⭐</text>
              <text x="110" y="22" text-anchor="middle" font-family="Bangers,cursive" font-size="14" fill="#FFE566">WINNING ALWAYS 😌</text>
            </svg>
          </div>
          <div class="gif-caption">😌 Current mood, permanently</div>
        </div>
        <div class="bubble-text">I don't lose vaad-vivaads. Ever. 😤</div>
        <div class="bubble-time">08:47 ✓✓</div>
      </div>
    </div>
  </div>
 
  <!-- 7. Short back and forth text chaos -->
  <div class="msg vini">
    <div class="avatar-sm">🦊</div>
    <div class="bubble-wrap">
      <div class="sender-name">Vini</div>
      <div class="bubble">
        <div class="bubble-text">You literally said "I'll prove it" in 47 consecutive messages 😐</div>
        <div class="bubble-time">08:48 ✓✓</div>
      </div>
    </div>
  </div>
 
  <div class="msg san">
    <div class="avatar-sm">🐱</div>
    <div class="bubble-wrap">
      <div class="sender-name">San (Billo Mausi)</div>
      <div class="bubble">
        <div class="bubble-text">That was VALID argumentation 😤😤</div>
        <div class="bubble-time">08:48 ✓✓</div>
      </div>
    </div>
  </div>
 
  <div class="msg vini">
    <div class="avatar-sm">🦊</div>
    <div class="bubble-wrap">
      <div class="sender-name">Vini</div>
      <div class="bubble">
        <div class="gif-panel">
          <div class="gif-badge">GIF</div>
          <div class="gif-canvas" style="background:linear-gradient(135deg,#37474F,#546E7A)">
            <svg width="220" height="155" viewBox="0 0 220 155">
              <!-- Three question marks -->
              <text x="60" y="90" font-family="Bangers,cursive" font-size="60" fill="#FFE566" opacity="0.9"
                style="animation:ha-pop 1.5s infinite">?</text>
              <text x="90" y="80" font-family="Bangers,cursive" font-size="44" fill="#FF7F00" opacity="0.8"
                style="animation:ha-pop 1.5s 0.5s infinite">?</text>
              <text x="114" y="95" font-family="Bangers,cursive" font-size="52" fill="#FFE566" opacity="0.9"
                style="animation:ha-pop 1.5s 1s infinite">?</text>
              <!-- Orange cat face just staring -->
              <g style="animation:body-bounce 2s infinite ease-in-out; transform-origin:160px 80px">
                <circle cx="160" cy="80" r="28" fill="#FF7F00"/>
                <polygon points="138,65 145,44 155,65" fill="#FF7F00"/>
                <polygon points="165,65 175,44 182,65" fill="#FF7F00"/>
                <polygon points="141,64 145,48 153,64" fill="#FFB3B3" opacity="0.6"/>
                <polygon points="167,64 175,48 180,64" fill="#FFB3B3" opacity="0.6"/>
                <!-- FLAT eyes -->
                <rect x="147" y="76" width="10" height="4" rx="2" fill="#333"/>
                <rect x="163" y="76" width="10" height="4" rx="2" fill="#333"/>
                <!-- Flat line mouth -->
                <line x1="153" y1="90" x2="167" y2="90" stroke="#333" stroke-width="2.5" stroke-linecap="round"/>
                <line x1="138" y1="78" x2="147" y2="79" stroke="#fff" stroke-width="1" opacity="0.7"/>
                <line x1="173" y1="79" x2="182" y2="78" stroke="#fff" stroke-width="1" opacity="0.7"/>
              </g>
            </svg>
          </div>
          <div class="gif-caption">🤨 Sir this was a 4-hour debate about dal</div>
        </div>
        <div class="bubble-time">08:49 ✓✓</div>
      </div>
    </div>
  </div>
 
  <!-- THEN — SURPRISE! Birthday message -->
  <div class="date-sep"><div class="date-pill" style="background:rgba(37,211,102,0.25); color:#075E54;">💐 AND THEN...</div></div>
 
  <!-- The actual Birthday GIF -->
  <div class="msg vini">
    <div class="avatar-sm" style="width:42px;height:42px;font-size:22px">🦊</div>
    <div class="bubble-wrap" style="max-width:min(88%,440px)">
      <div class="sender-name">Vini 🎂</div>
      <div class="bubble" style="background:linear-gradient(135deg,#DCF8C6,#B2F5C8)">
        <div class="gif-panel">
          <div class="gif-badge">GIF</div>
          <div class="gif-canvas" style="background:linear-gradient(135deg,#1a1a2e,#16213e); height:200px">
            <!-- Birthday celebration SVG -->
            <svg width="260" height="196" viewBox="0 0 260 196" style="overflow:visible">
              <!-- Confetti particles -->
              <g style="animation:confetti-fall 1.8s infinite linear">
                <rect x="30" y="-20" width="8" height="14" rx="2" fill="#FF006E" transform="rotate(20,34,-13)"/>
                <rect x="70" y="-30" width="6" height="10" rx="2" fill="#FFE566"/>
                <rect x="120" y="-10" width="7" height="12" rx="2" fill="#4CC9F0" transform="rotate(-15,123,-4)"/>
                <rect x="180" y="-25" width="8" height="13" rx="2" fill="#06D6A0" transform="rotate(30,184,-18)"/>
                <rect x="220" y="-15" width="6" height="11" rx="2" fill="#FF9F1C"/>
                <circle cx="50" cy="-18" r="5" fill="#7209B7"/>
                <circle cx="155" cy="-22" r="4" fill="#F72585"/>
                <circle cx="200" cy="-10" r="5" fill="#FFE566"/>
              </g>
              <g style="animation:confetti-fall 1.8s 0.6s infinite linear">
                <rect x="20" y="-30" width="7" height="11" rx="2" fill="#4CC9F0" transform="rotate(-20)"/>
                <rect x="90" y="-20" width="8" height="13" rx="2" fill="#FF9F1C" transform="rotate(15)"/>
                <rect x="140" y="-35" width="6" height="10" rx="2" fill="#F72585"/>
                <rect x="195" y="-15" width="9" height="14" rx="2" fill="#06D6A0" transform="rotate(-25)"/>
                <circle cx="110" cy="-25" r="5" fill="#FFE566"/>
                <circle cx="240" cy="-20" r="4" fill="#FF006E"/>
              </g>
              <g style="animation:confetti-fall 1.8s 1.2s infinite linear">
                <rect x="45" y="-25" width="6" height="10" rx="2" fill="#7209B7"/>
                <rect x="160" y="-30" width="8" height="13" rx="2" fill="#FF006E" transform="rotate(10)"/>
                <rect x="230" y="-20" width="7" height="12" rx="2" fill="#FFE566"/>
                <circle cx="85" cy="-18" r="4" fill="#06D6A0"/>
                <circle cx="210" cy="-22" r="5" fill="#4CC9F0"/>
              </g>
 
              <!-- Happy Vini cat dancing with banner -->
              <g style="animation:happy-sway 0.7s infinite ease-in-out; transform-origin:80px 110px">
                <!-- Body -->
                <ellipse cx="80" cy="140" rx="25" ry="20" fill="#FF7F00"/>
                <!-- Legs dancing -->
                <rect x="66" y="155" width="10" height="18" rx="5" fill="#FF6500" transform="rotate(-15,71,164)"/>
                <rect x="84" y="155" width="10" height="18" rx="5" fill="#FF6500" transform="rotate(20,89,164)"/>
                <!-- Arms holding banner -->
                <path d="M 100 130 L 140 118" stroke="#FF6500" stroke-width="8" fill="none" stroke-linecap="round"/>
                <path d="M 60 130 Q 42 120 38 108" stroke="#FF6500" stroke-width="8" fill="none" stroke-linecap="round"
                  style="animation:limb-flail 0.7s infinite; transform-origin:60px 130px"/>
                <!-- Head -->
                <circle cx="80" cy="106" r="27" fill="#FF7F00"/>
                <polygon points="57,92 65,68 75,92" fill="#FF7F00"/>
                <polygon points="85,92 95,68 103,92" fill="#FF7F00"/>
                <polygon points="60,91 65,72 73,91" fill="#FFB3B3" opacity="0.6"/>
                <polygon points="87,91 95,72 101,91" fill="#FFB3B3" opacity="0.6"/>
                <!-- Happy crescent eyes -->
                <path d="M 64 103 Q 72 98 80 103" stroke="#1a1a1a" fill="none" stroke-width="3" stroke-linecap="round"/>
                <path d="M 80 103 Q 88 98 96 103" stroke="#1a1a1a" fill="none" stroke-width="3" stroke-linecap="round"/>
                <!-- Big smile -->
                <path d="M 66 114 Q 80 126 94 114" stroke="#1a1a1a" fill="none" stroke-width="2.5" stroke-linecap="round"/>
                <!-- Nose -->
                <ellipse cx="80" cy="110" rx="4" ry="3" fill="#FF1493"/>
                <!-- Whiskers -->
                <line x1="46" y1="106" x2="72" y2="108" stroke="#fff" stroke-width="1.2" opacity="0.7"/>
                <line x1="88" y1="108" x2="114" y2="106" stroke="#fff" stroke-width="1.2" opacity="0.7"/>
                <!-- Rosy cheeks -->
                <ellipse cx="63" cy="111" rx="6" ry="4" fill="#FFB3B3" opacity="0.5"/>
                <ellipse cx="97" cy="111" rx="6" ry="4" fill="#FFB3B3" opacity="0.5"/>
              </g>
 
              <!-- Banner (held by Vini cat) -->
              <g style="animation:banner-wave 0.7s infinite ease-in-out; transform-origin:185px 118px">
                <rect x="136" y="108" width="98" height="34" rx="8" fill="#FFE566" stroke="#FF4500" stroke-width="2.5"/>
                <text x="185" y="130" text-anchor="middle" font-family="Bangers,cursive" font-size="17" fill="#FF4500" letter-spacing="1">BILLO MAUSI!</text>
              </g>
 
              <!-- Happy San cat on right -->
              <g style="animation:body-bounce 0.8s 0.4s infinite ease-in-out; transform-origin:200px 110px">
                <ellipse cx="200" cy="140" rx="22" ry="18" fill="#9B59B6"/>
                <!-- Birthday hat -->
                <polygon points="200,60 185,90 215,90" fill="#F72585"/>
                <circle cx="200" cy="60" r="4" fill="#FFE566"/>
                <line x1="185" y1="90" x2="215" y2="90" stroke="#FFE566" stroke-width="2"/>
                <!-- Head -->
                <circle cx="200" cy="104" r="25" fill="#9B59B6"/>
                <polygon points="178,90 185,68 196,90" fill="#9B59B6"/>
                <polygon points="204,90 215,68 222,90" fill="#9B59B6"/>
                <polygon points="181,89 185,72 194,89" fill="#CE93D8" opacity="0.7"/>
                <polygon points="206,89 215,72 220,89" fill="#CE93D8" opacity="0.7"/>
                <!-- Star eyes -->
                <text x="188" y="110" font-size="12" fill="#FFE566">⭐</text>
                <text x="204" y="110" font-size="12" fill="#FFE566">⭐</text>
                <!-- Big smile -->
                <path d="M 186 118 Q 200 130 214 118" stroke="#fff" fill="none" stroke-width="2.5" stroke-linecap="round"/>
                <line x1="170" y1="104" x2="190" y2="106" stroke="#fff" stroke-width="1.2" opacity="0.7"/>
                <line x1="210" y1="106" x2="230" y2="104" stroke="#fff" stroke-width="1.2" opacity="0.7"/>
                <!-- Arms raised -->
                <path d="M 178 125 Q 162 112 160 100" stroke="#8E44AD" stroke-width="7" fill="none" stroke-linecap="round"/>
                <path d="M 222 125 Q 238 112 240 100" stroke="#8E44AD" stroke-width="7" fill="none" stroke-linecap="round"/>
              </g>
 
              <!-- Floating hearts -->
              <text x="25" y="80" font-size="18" style="animation:heart-float 1.5s infinite">💛</text>
              <text x="235" y="70" font-size="16" style="animation:heart-float 1.5s 0.5s infinite">💜</text>
              <text x="120" y="175" font-size="20" style="animation:heart-float 1.5s 0.8s infinite">🎂</text>
 
              <!-- Glow behind cats -->
              <ellipse cx="140" cy="140" rx="120" ry="40" fill="#FFE566" opacity="0.08"
                style="animation:glow-pulse 1.5s infinite"/>
            </svg>
          </div>
          <div class="gif-caption">🎉 US on your birthday — me AND you celebrating YOU!</div>
        </div>
        <div class="bubble-text" style="font-size:16px; margin-top:6px; font-weight:700">
          Happy birthday billo mausiiiii!!! 🎂🎈<br>
          Uncalm, sweet, vaad-vivaad master 😤💛<br><br>
          From your favourite sparring partner,<br>— Vini 🦊
        </div>
        <div class="bubble-time">08:50 ✓✓</div>
      </div>
    </div>
  </div>
 
  <!-- San's reaction -->
  <div class="msg san">
    <div class="avatar-sm">🐱</div>
    <div class="bubble-wrap">
      <div class="sender-name">San (Billo Mausi)</div>
      <div class="bubble">
        <span class="reaction">😭🥺💛😤🥰😂</span>
        <div class="bubble-text">Lets vaad-vivaad and unleash the Chaos 😤😤</div>
        <div class="bubble-time">08:51 ✓✓</div>
      </div>
    </div>
  </div>
 
  <div class="msg vini">
    <div class="avatar-sm">🦊</div>
    <div class="bubble-wrap">
      <div class="sender-name">Vini</div>
      <div class="bubble">
        <span class="reaction">😹🥊</span>
        <div class="bubble-time">08:51 ✓✓</div>
      </div>
    </div>
  </div>
 
</div>
<!-- end chat -->
 
<!-- Big finale birthday card -->
<div class="bday-finale">
  <!-- Birthday cake SVG -->
  <svg width="90" height="95" viewBox="0 0 90 95" style="margin:0 auto 12px;display:block;position:relative;z-index:1">
    <rect x="20" y="48" width="50" height="30" rx="8" fill="#F72585"/>
    <rect x="8" y="60" width="74" height="28" rx="8" fill="#7209B7"/>
    <!-- Frosting -->
    <ellipse cx="35" cy="48" rx="8" ry="4" fill="white" opacity="0.7"/>
    <ellipse cx="55" cy="48" rx="8" ry="4" fill="white" opacity="0.7"/>
    <ellipse cx="22" cy="60" rx="7" ry="4" fill="white" opacity="0.6"/>
    <ellipse cx="45" cy="60" rx="7" ry="4" fill="white" opacity="0.6"/>
    <ellipse cx="68" cy="60" rx="7" ry="4" fill="white" opacity="0.6"/>
    <!-- Candles -->
    <rect x="32" y="34" width="8" height="18" rx="3" fill="#FF006E"/>
    <rect x="50" y="28" width="8" height="24" rx="3" fill="#FFE566"/>
    <!-- Flames -->
    <ellipse cx="36" cy="31" rx="5" ry="7" fill="#FFE566" style="animation:pow-pop 0.5s infinite"/>
    <ellipse cx="54" cy="25" rx="5" ry="7" fill="#FF9F1C" style="animation:pow-pop 0.5s 0.25s infinite"/>
    <!-- Plate -->
    <ellipse cx="45" cy="90" rx="42" ry="7" fill="#FFE566" opacity="0.8"/>
  </svg>
 
  <div class="bday-title">HAPPY BIRTHDAY<br>BILLO MAUSI!!! 🐱</div>
  <div class="bday-subtitle">UNCALM · SWEET · VAAD-VIVAAD MASTER</div>
  <div class="bday-msg">
    "To the one who fights me on WhatsApp with cat GIFs at 3am, who never backs down from a vaad-vivaad, who is somehow the sweetest and most uncalm person simultaneously — Happy Birthday San! 🥂💛"
  </div>
  <div class="friendship-badge">
    <svg width="28" height="28" viewBox="0 0 28 28">
      <circle cx="14" cy="14" r="13" fill="#FF7F00" stroke="none"/>
      <text x="14" y="19" text-anchor="middle" font-size="14">🦊</text>
    </svg>
    TOM & JERRY FRIENDSHIP — EST. MTECH DAYS
    <svg width="28" height="28" viewBox="0 0 28 28">
      <circle cx="14" cy="14" r="13" fill="#9B59B6" stroke="none"/>
      <text x="14" y="19" text-anchor="middle" font-size="14">🐱</text>
    </svg>
  </div>
  <div class="from-tag">FROM VINI — WITH ALL THE CHAOTIC LOVE 🥊💛</div>
</div>
 
<script>
// Subtle confetti on load
const body = document.body;
const confettiColors = ['#FF006E','#FFE566','#4CC9F0','#06D6A0','#FF9F1C','#7209B7','#F72585'];
function spawnConfetti() {
  const c = document.createElement('div');
  const size = Math.random() * 10 + 5;
  c.style.cssText = `
    position:fixed; top:-20px; left:${Math.random()*100}vw;
    width:${size}px; height:${size * (Math.random()>0.5 ? 0.4 : 1)}px;
    background:${confettiColors[Math.floor(Math.random()*confettiColors.length)]};
    border-radius:${Math.random()>0.5 ? '50%' : '2px'};
    pointer-events:none; z-index:9999;
    animation:confetti-fall ${2+Math.random()*2}s ${Math.random()*3}s linear forwards;
    transform:rotate(${Math.random()*360}deg);
  `;
  body.appendChild(c);
  setTimeout(() => c.remove(), 5000);
}
// Spawn confetti periodically
setInterval(spawnConfetti, 200);
for(let i=0;i<20;i++) setTimeout(spawnConfetti, i*100);
</script>
</body>
</html>
"""

# Render the HTML in your Streamlit app
# Set the height large enough to avoid internal scrolling for the chat interface
components.html(vini_html_content, height=1200, scrolling=True)