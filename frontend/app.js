const WS_URL = "ws://localhost:8001/ws/call";
const TARGET_SAMPLE_RATE = 16000;
const STREAM_MIME = "audio/mpeg";

const callBtn = document.getElementById("callBtn");
const statusEl = document.getElementById("status");
const logEl = document.getElementById("log");

let ws = null;

let audioContext = null;
let mediaStream = null;
let sourceNode = null;
let processorNode = null;
let silentGainNode = null;

let isCallActive = false;
let isConnecting = false;
let canSendMic = false;

let greetingAudio = null;

let legacyAudio = null;
let legacyAudioUrl = null;

let streamAudio = null;
let streamMediaSource = null;
let streamSourceBuffer = null;
let streamObjectUrl = null;
let streamChunks = [];
let streamEnded = false;
let streamProtocolActive = false;
let fallbackChunks = [];

let captionChars = [];
let captionStarts = [];
let captionDurations = [];
let captionLastEndMs = 0;
let captionShown = 0;
let captionLine = null;
let captionFrame = null;
let captionAudio = null;


function log(text, cls = "system") {
  const line = document.createElement("div");

  line.className = `log-line ${cls}`;
  line.textContent = text;

  logEl.appendChild(line);
  logEl.scrollTop = logEl.scrollHeight;
}


function floatTo16BitPCM(input) {
  const output = new Int16Array(input.length);

  for (let i = 0; i < input.length; i++) {
    const sample = Math.max(
      -1,
      Math.min(1, input[i])
    );

    output[i] =
      sample < 0
        ? Math.round(sample * 0x8000)
        : Math.round(sample * 0x7fff);
  }

  return output;
}


function resampleAudio(
  input,
  inputRate,
  outputRate
) {
  if (inputRate === outputRate) {
    return input.slice();
  }

  const ratio =
    inputRate / outputRate;

  const length =
    Math.round(input.length / ratio);

  const output =
    new Float32Array(length);

  for (let i = 0; i < length; i++) {
    const position = i * ratio;

    const index =
      Math.floor(position);

    const next =
      Math.min(
        index + 1,
        input.length - 1
      );

    const fraction =
      position - index;

    output[i] =
      input[index] * (1 - fraction) +
      input[next] * fraction;
  }

  return output;
}


function setListening() {
  if (
    !isCallActive ||
    !ws ||
    ws.readyState !== WebSocket.OPEN
  ) {
    return;
  }

  canSendMic = true;
  statusEl.textContent = "Listening...";
}


function setAgentBusy(
  text = "Agent responding..."
) {
  canSendMic = false;
  statusEl.textContent = text;
}


function stopCaptionSync() {
  if (captionFrame !== null) {
    cancelAnimationFrame(captionFrame);
    captionFrame = null;
  }

  captionAudio = null;
}


function resetCaptionForReply() {
  stopCaptionSync();

  captionChars = [];
  captionStarts = [];
  captionDurations = [];

  captionLastEndMs = 0;
  captionShown = 0;
  captionLine = null;
}


function ensureCaptionLine() {
  if (captionLine) {
    return captionLine;
  }

  captionLine =
    document.createElement("div");

  captionLine.className =
    "log-line agent";

  captionLine.textContent =
    "Agent: ";

  logEl.appendChild(captionLine);
  logEl.scrollTop = logEl.scrollHeight;

  return captionLine;
}


function addAlignment(alignment) {
  if (!alignment) {
    return;
  }

  const chars =
    alignment.chars || [];

  const starts =
    alignment.char_start_times_ms || [];

  const durations =
    alignment.char_durations_ms ||
    alignment.chars_durations_ms ||
    [];

  if (
    !chars.length ||
    !starts.length
  ) {
    return;
  }

  let offset = 0;

  if (captionStarts.length > 0) {
    const firstStart =
      Number(starts[0]) || 0;

    if (
      firstStart <
      Math.max(
        0,
        captionLastEndMs - 50
      )
    ) {
      offset = captionLastEndMs;
    }
  }

  const count =
    Math.min(
      chars.length,
      starts.length
    );

  for (let i = 0; i < count; i++) {
    const char =
      String(chars[i] ?? "");

    const rawStart =
      Number(starts[i]) || 0;

    const duration =
      Number(durations[i]) || 0;

    const start =
      rawStart + offset;

    captionChars.push(char);
    captionStarts.push(start);
    captionDurations.push(duration);

    captionLastEndMs =
      Math.max(
        captionLastEndMs,
        start + duration
      );
  }
}


function updateCaption() {
  if (!captionAudio) {
    return;
  }

  const currentMs =
    captionAudio.currentTime * 1000;

  while (
    captionShown < captionChars.length &&
    captionStarts[captionShown] <= currentMs
  ) {
    captionShown++;
  }

  if (captionShown > 0) {
    const line =
      ensureCaptionLine();

    line.textContent =
      "Agent: " +
      captionChars
        .slice(0, captionShown)
        .join("");

    logEl.scrollTop =
      logEl.scrollHeight;
  }

  if (
    captionAudio &&
    !captionAudio.ended &&
    isCallActive
  ) {
    captionFrame =
      requestAnimationFrame(
        updateCaption
      );
  }
}


function startCaptionSync(audio) {
  stopCaptionSync();

  captionAudio = audio;

  captionFrame =
    requestAnimationFrame(
      updateCaption
    );
}


function finishCaptionSync() {
  stopCaptionSync();

  if (!captionChars.length) {
    return;
  }

  captionShown =
    captionChars.length;

  const line =
    ensureCaptionLine();

  line.textContent =
    "Agent: " +
    captionChars.join("");

  logEl.scrollTop =
    logEl.scrollHeight;
}


function stopGreeting() {
  if (!greetingAudio) {
    return;
  }

  greetingAudio.onended = null;
  greetingAudio.onerror = null;

  try {
    greetingAudio.pause();
    greetingAudio.currentTime = 0;
  } catch (_) {}

  greetingAudio.src = "";
  greetingAudio = null;
}


function stopLegacyAudio() {
  if (legacyAudio) {
    legacyAudio.onended = null;
    legacyAudio.onerror = null;

    try {
      legacyAudio.pause();
      legacyAudio.currentTime = 0;
    } catch (_) {}

    legacyAudio.src = "";
    legacyAudio = null;
  }

  if (legacyAudioUrl) {
    URL.revokeObjectURL(
      legacyAudioUrl
    );

    legacyAudioUrl = null;
  }
}


function cleanupStreamAudio() {
  if (streamSourceBuffer) {
    try {
      streamSourceBuffer
        .removeEventListener(
          "updateend",
          pumpStream
        );
    } catch (_) {}
  }

  if (streamAudio) {
    streamAudio.onended = null;
    streamAudio.onerror = null;
    streamAudio.onplaying = null;

    try {
      streamAudio.pause();
      streamAudio.removeAttribute(
        "src"
      );
      streamAudio.load();
    } catch (_) {}

    streamAudio = null;
  }

  if (streamObjectUrl) {
    URL.revokeObjectURL(
      streamObjectUrl
    );

    streamObjectUrl = null;
  }

  streamMediaSource = null;
  streamSourceBuffer = null;

  streamChunks = [];
  fallbackChunks = [];

  streamEnded = false;
  streamProtocolActive = false;
}


function stopAllAgentAudio(
  resumeListening = false
) {
  stopCaptionSync();

  stopGreeting();
  stopLegacyAudio();
  cleanupStreamAudio();

  if (resumeListening) {
    setListening();
  }
}


function supportsStreamingAudio() {
  return (
    "MediaSource" in window &&
    MediaSource.isTypeSupported(
      STREAM_MIME
    )
  );
}


function playLegacyAudio(
  arrayBuffer
) {
  stopLegacyAudio();

  setAgentBusy(
    "Agent speaking..."
  );

  const blob =
    new Blob(
      [arrayBuffer],
      { type: STREAM_MIME }
    );

  legacyAudioUrl =
    URL.createObjectURL(blob);

  legacyAudio =
    new Audio(legacyAudioUrl);

  legacyAudio.preload = "auto";

  legacyAudio.onplaying = () => {
    startCaptionSync(
      legacyAudio
    );
  };

  legacyAudio.onended = () => {
    finishCaptionSync();

    stopLegacyAudio();
    setListening();
  };

  legacyAudio.onerror = () => {
    stopCaptionSync();

    log(
      "Agent audio playback failed.",
      "system"
    );

    stopLegacyAudio();
    setListening();
  };

  legacyAudio
    .play()
    .catch((err) => {
      stopCaptionSync();

      log(
        `Playback error: ${err.message}`,
        "system"
      );

      stopLegacyAudio();
      setListening();
    });
}


function pumpStream() {
  if (
    !streamSourceBuffer ||
    streamSourceBuffer.updating
  ) {
    return;
  }

  if (streamChunks.length > 0) {
    const chunk =
      streamChunks.shift();

    try {
      streamSourceBuffer
        .appendBuffer(chunk);
    } catch (err) {
      log(
        `Streaming audio error: ${err.message}`,
        "system"
      );
    }

    return;
  }

  if (
    streamEnded &&
    streamMediaSource &&
    streamMediaSource.readyState ===
      "open"
  ) {
    try {
      streamMediaSource
        .endOfStream();
    } catch (_) {}
  }
}


function startAudioStream() {
  stopLegacyAudio();
  cleanupStreamAudio();
  resetCaptionForReply();

  setAgentBusy(
    "Agent speaking..."
  );

  streamProtocolActive = true;
  streamEnded = false;

  if (!supportsStreamingAudio()) {
    log(
      "Streaming playback unavailable — using buffered playback.",
      "system"
    );

    return;
  }

  const mediaSource =
    new MediaSource();

  streamMediaSource =
    mediaSource;

  streamObjectUrl =
    URL.createObjectURL(
      mediaSource
    );

  streamAudio =
    new Audio();

  streamAudio.src =
    streamObjectUrl;

  streamAudio.preload =
    "auto";

  streamAudio.onplaying = () => {
    if (streamAudio) {
      startCaptionSync(
        streamAudio
      );
    }
  };

  streamAudio.onended = () => {
    finishCaptionSync();
    cleanupStreamAudio();
    setListening();
  };

  streamAudio.onerror = () => {
    stopCaptionSync();

    log(
      "Streaming audio playback failed.",
      "system"
    );

    cleanupStreamAudio();
    setListening();
  };

  mediaSource.addEventListener(
    "sourceopen",
    () => {
      if (
        streamMediaSource !==
          mediaSource ||
        mediaSource.readyState !==
          "open"
      ) {
        return;
      }

      try {
        streamSourceBuffer =
          mediaSource.addSourceBuffer(
            STREAM_MIME
          );

        streamSourceBuffer.mode =
          "sequence";

        streamSourceBuffer
          .addEventListener(
            "updateend",
            pumpStream
          );

        pumpStream();

        streamAudio
          ?.play()
          .catch((err) => {
            log(
              `Playback error: ${err.message}`,
              "system"
            );
          });

      } catch (err) {
        log(
          `Unable to initialize streaming audio: ${err.message}`,
          "system"
        );
      }
    },
    { once: true }
  );

  streamAudio
    .play()
    .catch(() => {});
}


function appendStreamAudio(
  arrayBuffer
) {
  if (!streamProtocolActive) {
    resetCaptionForReply();

    playLegacyAudio(
      arrayBuffer
    );

    return;
  }

  setAgentBusy(
    "Agent speaking..."
  );

  if (!supportsStreamingAudio()) {
    fallbackChunks.push(
      arrayBuffer
    );

    return;
  }

  streamChunks.push(
    new Uint8Array(
      arrayBuffer
    )
  );

  pumpStream();
}


function finishAudioStream() {
  if (!streamProtocolActive) {
    return;
  }

  if (!supportsStreamingAudio()) {
    const chunks =
      fallbackChunks.slice();

    fallbackChunks = [];
    streamProtocolActive = false;

    if (!chunks.length) {
      finishCaptionSync();
      setListening();
      return;
    }

    const blob =
      new Blob(
        chunks,
        { type: STREAM_MIME }
      );

    legacyAudioUrl =
      URL.createObjectURL(blob);

    legacyAudio =
      new Audio(
        legacyAudioUrl
      );

    legacyAudio.preload =
      "auto";

    legacyAudio.onplaying =
      () => {
        startCaptionSync(
          legacyAudio
        );
      };

    legacyAudio.onended =
      () => {
        finishCaptionSync();

        stopLegacyAudio();
        setListening();
      };

    legacyAudio.onerror =
      () => {
        stopCaptionSync();

        stopLegacyAudio();
        setListening();
      };

    legacyAudio
      .play()
      .catch((err) => {
        stopCaptionSync();

        log(
          `Playback error: ${err.message}`,
          "system"
        );

        stopLegacyAudio();
        setListening();
      });

    return;
  }

  streamEnded = true;
  pumpStream();
}


async function prepareAudioContext() {
  const AudioContextClass =
    window.AudioContext ||
    window.webkitAudioContext;

  if (!AudioContextClass) {
    throw new Error(
      "Web Audio API is not supported."
    );
  }

  if (!audioContext) {
    audioContext =
      new AudioContextClass();
  }

  if (
    audioContext.state ===
    "suspended"
  ) {
    await audioContext.resume();
  }

  log(
    `Browser audio: ${audioContext.sampleRate} Hz`,
    "system"
  );
}


async function startMic() {
  if (
    !navigator.mediaDevices
      ?.getUserMedia
  ) {
    throw new Error(
      "Microphone access is not supported."
    );
  }

  statusEl.textContent =
    "Requesting microphone permission...";

  mediaStream =
    await navigator.mediaDevices
      .getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1
        }
      });

  const tracks =
    mediaStream.getAudioTracks();

  if (!tracks.length) {
    throw new Error(
      "No microphone was detected."
    );
  }

  const track =
    tracks[0];

  const settings =
    track.getSettings?.() || {};

  log(
    `Microphone: ${
      track.label || "default"
    }`,
    "system"
  );

  if (settings.sampleRate) {
    log(
      `Microphone sample rate: ${settings.sampleRate} Hz`,
      "system"
    );
  }

  if (
    !audioContext ||
    audioContext.state === "closed"
  ) {
    await prepareAudioContext();
  }

  if (
    audioContext.state ===
    "suspended"
  ) {
    await audioContext.resume();
  }

  sourceNode =
    audioContext
      .createMediaStreamSource(
        mediaStream
      );

  processorNode =
    audioContext
      .createScriptProcessor(
        4096,
        1,
        1
      );

  silentGainNode =
    audioContext.createGain();

  silentGainNode.gain.value = 0;

  processorNode.onaudioprocess =
    (event) => {
      if (
        !isCallActive ||
        !canSendMic ||
        !ws ||
        ws.readyState !==
          WebSocket.OPEN
      ) {
        return;
      }

      const input =
        event.inputBuffer
          .getChannelData(0);

      if (!input?.length) {
        return;
      }

      const inputRate =
        event.inputBuffer
          .sampleRate ||
        audioContext.sampleRate;

      const resampled =
        resampleAudio(
          input,
          inputRate,
          TARGET_SAMPLE_RATE
        );

      const pcm =
        floatTo16BitPCM(
          resampled
        );

      if (
        pcm.length &&
        ws.readyState ===
          WebSocket.OPEN
      ) {
        ws.send(
          pcm.buffer
        );
      }
    };

  sourceNode.connect(
    processorNode
  );

  processorNode.connect(
    silentGainNode
  );

  silentGainNode.connect(
    audioContext.destination
  );

  log(
    `Microphone ready — ${TARGET_SAMPLE_RATE} Hz PCM16`,
    "system"
  );
}


function playGreeting() {
  stopAllAgentAudio(false);

  setAgentBusy(
    "Agent greeting..."
  );

  greetingAudio =
    new Audio(
      "/greeting.mp3"
    );

  greetingAudio.preload =
    "auto";

  greetingAudio.onended =
    () => {
      greetingAudio = null;

      log(
        "Greeting finished — listening",
        "system"
      );

      setListening();
    };

  greetingAudio.onerror =
    () => {
      log(
        "Greeting playback failed.",
        "system"
      );

      greetingAudio = null;
      setListening();
    };

  greetingAudio
    .play()
    .catch((err) => {
      log(
        `Greeting error: ${err.message}`,
        "system"
      );

      greetingAudio = null;
      setListening();
    });

  log(
    "Agent: Hi! I'm Vishal's AI assistant. How can I help you today?",
    "agent"
  );
}


function handleControlMessage(
  msg
) {
  switch (msg.type) {

    case "session_started":
      statusEl.textContent =
        `In call (${
          msg.session_id.slice(
            0,
            8
          )
        }...)`;

      log(
        "Call connected",
        "system"
      );

      break;


    case "play_greeting":
      playGreeting();
      break;


    case "transcript":
      log(
        `You: ${msg.text}`,
        "user"
      );

      break;


    case "reply_text":
      setAgentBusy(
        "Agent responding..."
      );

      break;


    case "audio_start":
      startAudioStream();
      break;


    case "audio_alignment":
      addAlignment(
        msg.alignment
      );

      break;


    case "audio_end":
      finishAudioStream();
      break;


    case "stop_audio":
      stopAllAgentAudio(
        true
      );

      break;


    case "error":
      log(
        `Server error: ${
          msg.message ||
          "Unknown error"
        }`,
        "system"
      );

      stopAllAgentAudio(
        true
      );

      break;


    case "audio_ack":
      break;


    default:
      log(
        `Unknown message: ${
          JSON.stringify(msg)
        }`,
        "system"
      );
  }
}


async function startCall() {
  if (
    isConnecting ||
    isCallActive
  ) {
    return;
  }

  isConnecting = true;
  canSendMic = false;

  callBtn.textContent =
    "Connecting...";

  callBtn.classList.add(
    "active"
  );

  statusEl.textContent =
    "Preparing audio...";

  try {
    await prepareAudioContext();
    await startMic();

    isCallActive = true;

    ws =
      new WebSocket(
        WS_URL
      );

    ws.binaryType =
      "arraybuffer";


    ws.onopen = () => {
      isConnecting = false;

      callBtn.textContent =
        "End Call";

      statusEl.textContent =
        "Connected — preparing agent...";

      log(
        "WebSocket connected",
        "system"
      );
    };


    ws.onmessage =
      (event) => {
        if (!isCallActive) {
          return;
        }

        if (
          typeof event.data ===
          "string"
        ) {
          try {
            handleControlMessage(
              JSON.parse(
                event.data
              )
            );

          } catch (err) {
            log(
              `Invalid server message: ${err.message}`,
              "system"
            );
          }

          return;
        }

        appendStreamAudio(
          event.data
        );
      };


    ws.onerror = () => {
      log(
        "WebSocket connection error.",
        "system"
      );
    };


    ws.onclose = () => {
      const wasActive =
        isCallActive ||
        isConnecting;

      cleanupCall(false);

      if (wasActive) {
        statusEl.textContent =
          "Disconnected";

        log(
          "WebSocket closed",
          "system"
        );
      }
    };

  } catch (err) {
    log(
      `Unable to start call: ${err.message}`,
      "system"
    );

    statusEl.textContent =
      "Could not start call";

    cleanupCall(true);
  }
}


function cleanupCall(
  closeSocket = true
) {
  canSendMic = false;
  isCallActive = false;
  isConnecting = false;

  stopCaptionSync();
  stopAllAgentAudio(false);

  if (processorNode) {
    processorNode.onaudioprocess =
      null;

    try {
      processorNode.disconnect();
    } catch (_) {}

    processorNode = null;
  }

  if (sourceNode) {
    try {
      sourceNode.disconnect();
    } catch (_) {}

    sourceNode = null;
  }

  if (silentGainNode) {
    try {
      silentGainNode.disconnect();
    } catch (_) {}

    silentGainNode = null;
  }

  if (mediaStream) {
    mediaStream
      .getTracks()
      .forEach(
        (track) => {
          try {
            track.stop();
          } catch (_) {}
        }
      );

    mediaStream = null;
  }

  if (audioContext) {
    const context =
      audioContext;

    audioContext = null;

    if (
      context.state !==
      "closed"
    ) {
      context
        .close()
        .catch(() => {});
    }
  }

  if (ws) {
    const socket = ws;

    ws = null;

    if (
      closeSocket &&
      (
        socket.readyState ===
          WebSocket.OPEN ||
        socket.readyState ===
          WebSocket.CONNECTING
      )
    ) {
      try {
        socket.close();
      } catch (_) {}
    }
  }

  resetCaptionForReply();

  callBtn.textContent =
    "Start Call";

  callBtn.classList.remove(
    "active"
  );
}


function stopCall() {
  const hadCall =
    isCallActive ||
    isConnecting;

  cleanupCall(true);

  statusEl.textContent =
    "Not connected";

  if (hadCall) {
    log(
      "Call ended",
      "system"
    );
  }
}


callBtn.addEventListener(
  "click",
  async () => {
    if (
      isCallActive ||
      isConnecting
    ) {
      stopCall();
      return;
    }

    logEl.innerHTML = "";

    await startCall();
  }
);


window.addEventListener(
  "beforeunload",
  () => cleanupCall(true)
);