#!/usr/bin/env node
const { readFileSync } = require("fs");
const { join } = require("path");

const DIR = join(__dirname, "claudeclaw");
const STATE_FILE = join(DIR, "state.json");
const PID_FILE = join(DIR, "daemon.pid");

const R = "\x1b[0m";
const DIM = "\x1b[2m";
const RED = "\x1b[31m";
const GREEN = "\x1b[32m";

function fmt(ms) {
  if (ms <= 0) return GREEN + "now!" + R;
  var s = Math.floor(ms / 1000);
  var h = Math.floor(s / 3600);
  var m = Math.floor((s % 3600) / 60);
  if (h > 0) return h + "h " + m + "m";
  if (m > 0) return m + "m";
  return (s % 60) + "s";
}

function alive() {
  try {
    var pid = readFileSync(PID_FILE, "utf-8").trim();
    process.kill(Number(pid), 0);
    return true;
  } catch { return false; }
}

var B = DIM + "\u2502" + R;
var TL = DIM + "\u256d" + R;
var TR = DIM + "\u256e" + R;
var BL = DIM + "\u2570" + R;
var BR = DIM + "\u256f" + R;
var H = DIM + "\u2500" + R;
var HEADER = TL + H.repeat(6) + " \ud83e\udd9e ClaudeClaw \ud83e\udd9e " + H.repeat(6) + TR;
var FOOTER = BL + H.repeat(30) + BR;

if (!alive()) {
  process.stdout.write(
    HEADER + "\n" +
    B + "        " + RED + "\u25cb offline" + R + "              " + B + "\n" +
    FOOTER
  );
  process.exit(0);
}

try {
  var state = JSON.parse(readFileSync(STATE_FILE, "utf-8"));
  var now = Date.now();
  var info = [];

  if (state.heartbeat) {
    info.push("\ud83d\udc93 " + fmt(state.heartbeat.nextAt - now));
  }

  var jc = (state.jobs || []).length;
  info.push("\ud83d\udccb " + jc + " job" + (jc !== 1 ? "s" : ""));
  info.push(GREEN + "\u25cf live" + R);

  if (state.telegram) {
    info.push(GREEN + "\ud83d\udce1" + R);
  }

  if (state.discord) {
    info.push(GREEN + "\ud83c\udfae" + R);
  }

  var mid = " " + info.join(" " + B + " ") + " ";

  process.stdout.write(HEADER + "\n" + B + mid + B + "\n" + FOOTER);
} catch {
  process.stdout.write(
    HEADER + "\n" +
    B + DIM + "         waiting...         " + R + B + "\n" +
    FOOTER
  );
}
