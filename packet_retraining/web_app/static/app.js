const csrfToken = document.querySelector('meta[name="csrf-token"]').content;
const posture = document.getElementById('posture');
const activityList = document.getElementById('activity-list');
const formMessage = document.getElementById('form-message');
const interfaceSelect = document.getElementById('interface-select');
const startButton = document.getElementById('start-button');
const stopButton = document.getElementById('stop-button');
const numberFormat = new Intl.NumberFormat();
let latestShownId = null;
let polling = false;
let activitySignature = '';

async function requestJson(url, options = {}) {
  const headers = {...options.headers};
  if (options.method === 'POST') headers['X-CSRF-Token'] = csrfToken;
  const response = await fetch(url, {...options, headers});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Request failed');
  return data;
}

function setMessage(message, success = false) {
  formMessage.textContent = message;
  formMessage.classList.toggle('success', success);
}

function formatPercent(score) {
  return `${(score * 100).toFixed(1)}%`;
}

function describeOrigin(result) {
  if (result.origin === 'live') return 'LIVE CAPTURE · EXPERIMENTAL';
  if (result.origin === 'pcap') return 'PCAP ANALYSIS · EXPERIMENTAL';
  return 'MANUAL FLOW';
}

function renderPosture(result) {
  latestShownId = result.id;
  posture.classList.toggle('is-attack', result.attack);
  posture.classList.toggle('is-normal', !result.attack);
  document.getElementById('posture-title').textContent = result.attack ? 'Intrusion signal flagged' : 'Flow appears normal';
  document.getElementById('posture-description').textContent = result.attack
    ? 'The model found a pattern associated with attack traffic. Review the flow details and investigate before taking action.'
    : 'This flow fell on the normal side of the model threshold. Continue monitoring for changes.';
  document.getElementById('posture-label').textContent = result.attack ? 'ATTACK PATTERN' : 'NORMAL PATTERN';
  document.getElementById('posture-origin').textContent = describeOrigin(result);
  document.getElementById('radar-icon').textContent = result.attack ? '!' : '✓';
  document.getElementById('radar-value').textContent = formatPercent(result.score);
  document.getElementById('radar-caption').textContent = result.attack ? 'ELEVATED RISK SIGNAL' : 'LOW RISK SIGNAL';
}

function describeEndpoint(result) {
  const endpoint = result.endpoints || {};
  if (!endpoint.source) return `${result.fields.proto.toUpperCase()} · Manual entry`;
  const sourcePort = endpoint.source_port ? `:${endpoint.source_port}` : '';
  const destinationPort = endpoint.destination_port ? `:${endpoint.destination_port}` : '';
  return `${endpoint.source}${sourcePort} → ${endpoint.destination}${destinationPort}`;
}

function renderActivity(events) {
  activityList.replaceChildren();
  if (!events.length) {
    const empty = document.createElement('div');
    empty.className = 'empty-activity';
    const orbit = document.createElement('span');
    orbit.className = 'empty-orbit';
    orbit.textContent = '◎';
    const title = document.createElement('strong');
    title.textContent = 'No signals yet';
    const description = document.createElement('p');
    description.textContent = 'Your analyzed flows and alerts will appear here.';
    empty.append(orbit, title, description);
    activityList.append(empty);
    return;
  }
  for (const result of events) {
    const item = document.createElement('button');
    item.type = 'button';
    item.className = `activity-item${result.attack ? ' is-attack' : ''}`;
    item.style.width = '100%';
    item.style.color = 'inherit';
    item.style.textAlign = 'left';
    item.style.cursor = 'pointer';
    const icon = document.createElement('span');
    icon.className = 'activity-icon';
    icon.textContent = result.attack ? '!' : '✓';
    const copy = document.createElement('span');
    copy.className = 'activity-copy';
    const title = document.createElement('strong');
    title.textContent = result.attack ? 'Attack signal flagged' : 'Normal flow';
    const subtitle = document.createElement('small');
    subtitle.textContent = describeEndpoint(result);
    copy.append(title, subtitle);
    const score = document.createElement('span');
    score.className = 'activity-score';
    score.textContent = formatPercent(result.score);
    const time = document.createElement('small');
    time.textContent = new Date(result.timestamp).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});
    score.append(time);
    item.append(icon, copy, score);
    item.addEventListener('click', () => renderPosture(result));
    activityList.append(item);
  }
}

function renderStatus(status) {
  document.getElementById('stat-total').textContent = numberFormat.format(status.total);
  document.getElementById('stat-attacks').textContent = numberFormat.format(status.attacks);
  document.getElementById('stat-normal').textContent = numberFormat.format(status.normal);
  document.getElementById('stat-capture').textContent = status.state === 'running' ? 'SCANNING' : 'OFFLINE';
  document.getElementById('stat-packets').textContent = `${numberFormat.format(status.packets)} packets observed`;
  document.getElementById('top-status').textContent = status.state === 'running' ? 'MONITORING LIVE' : 'SYSTEM READY';
  document.getElementById('capture-led').classList.toggle('active', status.state === 'running');
  document.getElementById('capture-message').textContent = status.error || (status.state === 'running' ? 'Live monitoring active' : 'Capture idle');
  startButton.disabled = status.state === 'running' || !status.tshark_available;
  stopButton.disabled = status.state !== 'running';
  const nextSignature = status.events.map(event => event.id).join(',');
  if (nextSignature !== activitySignature) {
    renderActivity(status.events);
    activitySignature = nextSignature;
  }
  if (status.events.length && status.events[0].id !== latestShownId) renderPosture(status.events[0]);
}

async function refreshStatus() {
  if (polling) return;
  polling = true;
  try {
    const status = await requestJson('/api/status');
    renderStatus(status);
  } catch (error) {
    document.getElementById('top-status').textContent = 'CONNECTION LOST';
  } finally {
    polling = false;
  }
}

async function loadConfig() {
  try {
    const config = await requestJson('/api/config');
    const note = document.getElementById('tshark-note');
    if (!config.tshark_available) {
      note.textContent = 'TShark is not installed. Install Wireshark with TShark and Npcap, then restart this app to enable PCAP and live capture.';
      note.classList.add('warning');
      startButton.disabled = true;
      return;
    }
    if (config.interface_error) {
      note.textContent = config.interface_error;
      note.classList.add('warning');
      return;
    }
    note.textContent = `${config.interfaces.length} interface${config.interfaces.length === 1 ? '' : 's'} available. Select the interface that carries the traffic you want to inspect.`;
    for (const item of config.interfaces) {
      const option = document.createElement('option');
      option.value = item.id;
      option.textContent = `${item.id}. ${item.name}`;
      interfaceSelect.append(option);
    }
  } catch (error) {
    const note = document.getElementById('tshark-note');
    note.textContent = error.message;
    note.classList.add('warning');
  }
}

for (const tab of document.querySelectorAll('.tab')) {
  tab.addEventListener('click', () => {
    for (const other of document.querySelectorAll('.tab')) {
      const active = other === tab;
      other.classList.toggle('active', active);
      other.setAttribute('aria-selected', String(active));
      const panel = document.getElementById(`panel-${other.dataset.tab}`);
      panel.classList.toggle('active', active);
      panel.hidden = !active;
    }
    setMessage('');
  });
}

const sampleFlows = {
  normal: {proto: 'tcp', dur: 0.005896, spkts: 22, dpkts: 24, sbytes: 2054, dbytes: 2478, sttl: 31, dttl: 29},
  attack: {proto: 'tcp', dur: 7.629429, spkts: 202, dpkts: 1294, sbytes: 9082, dbytes: 1747118, sttl: 62, dttl: 252},
};
for (const button of document.querySelectorAll('[data-sample]')) {
  button.addEventListener('click', () => {
    const form = document.getElementById('manual-form');
    for (const [name, value] of Object.entries(sampleFlows[button.dataset.sample])) form.elements[name].value = value;
    setMessage('Example loaded. Select Analyze flow to score it.', true);
  });
}

document.getElementById('manual-form').addEventListener('submit', async event => {
  event.preventDefault();
  const button = event.currentTarget.querySelector('button[type="submit"]');
  const data = new FormData(event.currentTarget);
  const fields = {proto: String(data.get('proto')).trim().toLowerCase()};
  for (const name of ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'sttl', 'dttl']) {
    fields[name] = Number(data.get(name));
  }
  button.disabled = true;
  setMessage('Analyzing flow…', true);
  try {
    const result = await requestJson('/api/predict', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({fields}),
    });
    renderPosture(result);
    setMessage(`Analysis complete · ${result.label} · ${formatPercent(result.score)} attack score`, true);
    await refreshStatus();
  } catch (error) {
    setMessage(error.message);
  } finally {
    button.disabled = false;
  }
});

const fileInput = document.getElementById('pcap-file');
const dropzone = document.querySelector('.dropzone');
fileInput.addEventListener('change', () => {
  document.getElementById('file-name').textContent = fileInput.files[0]?.name || 'Drop a capture here or browse files';
});
for (const eventName of ['dragenter', 'dragover']) {
  dropzone.addEventListener(eventName, event => { event.preventDefault(); dropzone.classList.add('dragover'); });
}
for (const eventName of ['dragleave', 'drop']) {
  dropzone.addEventListener(eventName, event => { event.preventDefault(); dropzone.classList.remove('dragover'); });
}
dropzone.addEventListener('drop', event => {
  if (event.dataTransfer.files.length) {
    fileInput.files = event.dataTransfer.files;
    fileInput.dispatchEvent(new Event('change'));
  }
});

document.getElementById('pcap-form').addEventListener('submit', async event => {
  event.preventDefault();
  if (!fileInput.files.length) return setMessage('Choose a capture file first.');
  const button = event.currentTarget.querySelector('button[type="submit"]');
  const body = new FormData();
  body.append('file', fileInput.files[0]);
  button.disabled = true;
  setMessage('Reading capture and scoring flows…', true);
  try {
    const summary = await requestJson('/api/pcap', {method: 'POST', body});
    const box = document.getElementById('upload-summary');
    box.hidden = false;
    box.textContent = `${numberFormat.format(summary.packets)} packets read · ${numberFormat.format(summary.flows)} flows scored · ${numberFormat.format(summary.attacks)} attack flags · ${numberFormat.format(summary.normal)} normal`;
    if (summary.results.length) renderPosture(summary.results[0]);
    setMessage('Capture analysis complete.', true);
    await refreshStatus();
  } catch (error) {
    setMessage(error.message);
  } finally {
    button.disabled = false;
  }
});

startButton.addEventListener('click', async () => {
  if (!interfaceSelect.value) return setMessage('Choose a network interface first.');
  startButton.disabled = true;
  setMessage('Starting local capture…', true);
  try {
    await requestJson('/api/live/start', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({interface: interfaceSelect.value}),
    });
    setMessage('Live capture started. New flow results will appear above.', true);
    await refreshStatus();
  } catch (error) {
    setMessage(error.message);
    startButton.disabled = false;
  }
});

stopButton.addEventListener('click', async () => {
  stopButton.disabled = true;
  try {
    await requestJson('/api/live/stop', {method: 'POST'});
    setMessage('Live capture stopped.', true);
    await refreshStatus();
  } catch (error) {
    setMessage(error.message);
    stopButton.disabled = false;
  }
});

const revealItems = document.querySelectorAll('.reveal');
if ('IntersectionObserver' in window) {
  const revealObserver = new IntersectionObserver(entries => {
    for (const entry of entries) {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-visible');
        revealObserver.unobserve(entry.target);
      }
    }
  }, {threshold: 0.08});
  for (const item of revealItems) revealObserver.observe(item);
} else {
  for (const item of revealItems) item.classList.add('is-visible');
}

loadConfig();
refreshStatus();
setInterval(refreshStatus, 2000);
