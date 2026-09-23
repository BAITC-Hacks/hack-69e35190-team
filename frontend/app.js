'use strict';
(() => {
  const el = id => document.getElementById(id);
  const escapeHtml = value => String(value).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  const money = value => Number(value).toLocaleString('ru-RU') + ' ₸';
  const date = value => value.split('-').reverse().join('.');
  let slots = {}, serial = 0, busy = false, datasets = [], datasetId = 'default';

  async function api(path, body) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 12000);
    try {
      const response = await fetch(path, {
        method: body === undefined ? 'GET' : 'POST',
        headers: {'Content-Type':'application/json'},
        body: body === undefined ? undefined : JSON.stringify(body), signal:controller.signal
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || `Ошибка сервера: HTTP ${response.status}`);
      return result;
    } catch (error) {
      if (error.name === 'AbortError') throw new Error('Сервер не ответил за 12 секунд. Повторите запрос');
      if (error instanceof TypeError) throw new Error('Нет связи с Python-сервером. Запустите python3 -m backend.api и откройте http://127.0.0.1:8000');
      if (error instanceof SyntaxError) throw new Error('Сервер вернул не JSON. Откройте интерфейс через python3 -m backend.api');
      throw error;
    } finally { clearTimeout(timer); }
  }

  function add(role, html) {
    const row = document.createElement('div'); row.className = 'msg-row ' + role;
    const bubble = document.createElement('div'); bubble.className = 'bubble'; bubble.innerHTML = html;
    row.appendChild(bubble); el('chat').appendChild(row); el('chat').scrollTop = el('chat').scrollHeight;
  }

  function summary() {
    return [slots.category, slots.city, slots.event_type, slots.event_date && date(slots.event_date),
      slots.budget_kzt && money(slots.budget_kzt), slots.duration_hours && `${slots.duration_hours} ч`, slots.language].filter(Boolean).join(' · ');
  }

  function resultHTML(result) {
    const titles = {matched:'Подобрали подрядчиков.',category_not_in_city:'Нет такой категории в городе.',no_matches:'Кандидаты есть, но ни один не подходит.'};
    const style = {matched:'matched',category_not_in_city:'warn',no_matches:'bad'}[result.status];
    let html = `<div class="outcome-banner ${style}"><b>${titles[result.status]}</b><br>${escapeHtml(result.message)}</div>`;
    for (const [i,p] of result.cards.entries()) {
      const tags = [p.synthetic ? 'синтетический профиль' : 'несинтетический профиль (по CSV)'];
      if (datasetId !== 'default') tags.push('из загруженного CSV');
      if (p.city_imputed) tags.push('город проставлен при подготовке');
      if (p.price_imputed) tags.push('цену «от» нужно уточнить');
      html += `<div class="card"><div class="card-top"><div><span class="card-rank">#${i+1}</span><span class="card-name">${escapeHtml(p.anon_name)}</span></div><div class="card-price">от ${money(p.price_from_kzt)}</div></div><div class="card-cat">${escapeHtml(p.category)} — ${escapeHtml(p.city)}</div><div class="card-tags">${tags.map(t=>`<span class="tag">${escapeHtml(t)}</span>`).join('')}</div><div class="card-explain">${escapeHtml(p.explanation)}</div></div>`;
    }
    return html;
  }

  function chips(values) {
    el('chips').replaceChildren();
    for (const value of values || []) {
      const button = document.createElement('button'); button.className = 'chip'; button.textContent = value;
      button.addEventListener('click', () => send(value)); el('chips').appendChild(button);
    }
  }

  function reset() {
    serial++; slots = {}; busy = false; el('sendBtn').disabled = false; el('sendBtn').textContent = 'Отправить';
    el('chat').replaceChildren(); chips([]);
    const selected = datasets.find(d => d.id === datasetId);
    add('bot', 'Опишите мероприятие: город, дата, тип, категория и бюджет; необязательно — язык и часы. Например: «фотосессия на корпоратив в Алмате 25 октября на 3 часа, бюджет 50тыс».');
    if (selected) {
      add('bot', `Каталог: ${escapeHtml(selected.name)}. Календарь: ${date(selected.calendar_start)} — ${date(selected.calendar_end)}. В каком городе ищем?`);
      chips(selected.cities);
    }
  }

  async function send(text) {
    if (!text.trim() || busy) return;
    if (/^(сброс|заново|новый поиск|начать заново)$/i.test(text.trim())) { reset(); return; }
    add('user', escapeHtml(text));
    const current = ++serial; busy = true; el('sendBtn').disabled = true; el('sendBtn').textContent = 'Подбираю…'; chips([]);
    try {
      const response = await api('/chat', {message:text, slots, dataset_id:datasetId});
      if (current !== serial) return;
      slots = response.slots;
      let html = `<div class="slots-note">Распознано: ${escapeHtml(summary()) || 'пока нет параметров'}</div>`;
      for (const warning of response.warnings) html += `<p>${escapeHtml(warning)}</p>`;
      if (response.result) html += resultHTML(response.result);
      else { html += escapeHtml(response.prompt); chips(response.choices); }
      add('bot', html);
    } catch (error) {
      if (current === serial) add('bot', `<div class="outcome-banner bad">${escapeHtml(error.message)}</div>`);
    } finally {
      if (current === serial) { busy = false; el('sendBtn').disabled = false; el('sendBtn').textContent = 'Отправить'; }
    }
  }

  function showDataset() {
    const d = datasets.find(d => d.id === datasetId);
    if (!d) return;
    const status = {ready:'семантический индекс готов',lexical:'локальный поиск',indexing:'индекс строится',degraded:'API недоступен — локальный поиск'}[d.index_status];
    el('datasetInfo').textContent = `${d.profile_count} профилей, из них синтетических ${d.synthetic_count}. ${status}. ${d.index_message}`;
    el('presets').hidden = datasetId !== 'default';
  }

  async function refresh(selectedId) {
    datasets = (await api('/datasets')).datasets;
    if (selectedId) datasetId = selectedId;
    el('datasetSelect').replaceChildren();
    for (const d of datasets) {
      const option = document.createElement('option'); option.value = d.id; option.textContent = d.name;
      el('datasetSelect').appendChild(option);
    }
    el('datasetSelect').value = datasetId; showDataset();
  }

  el('sendBtn').addEventListener('click', () => { if (busy) return; const value = el('msgInput').value; el('msgInput').value = ''; send(value); });
  el('msgInput').addEventListener('keydown', event => { if (event.key === 'Enter') { event.preventDefault(); el('sendBtn').click(); } });
  el('resetBtn').addEventListener('click', reset);
  el('datasetSelect').addEventListener('change', () => { datasetId = el('datasetSelect').value; showDataset(); reset(); });
  el('refreshDatasets').addEventListener('click', () => refresh().catch(error => { el('datasetInfo').textContent = error.message; }));
  el('indexDataset').addEventListener('click', async () => {
    try { await api('/datasets/index', {dataset_id:datasetId}); await refresh(); }
    catch (error) { el('datasetInfo').textContent = error.message; }
  });
  el('uploadForm').addEventListener('submit', async event => {
    event.preventDefault(); el('uploadBtn').disabled = true;
    try {
      const file = el('datasetFile').files[0];
      if (!file || file.size > 2000000) throw new Error('Выберите CSV до 2 МБ');
      el('uploadStatus').textContent = 'Проверяем CSV…';
      const result = await api('/datasets', {name:el('datasetName').value, csv:await file.text(), calendar_start:el('calendarStart').value, calendar_end:el('calendarEnd').value});
      await refresh(result.id); reset(); el('uploadStatus').textContent = 'Каталог загружен и выбран. Подбор уже доступен; статус AI-индекса можно обновить.';
    } catch (error) { el('uploadStatus').textContent = error.message; }
    finally { el('uploadBtn').disabled = false; }
  });
  const demos = {
    dense:'Корпоратив в Алматы 10 октября, ведущий, бюджет 1500000',
    date:'Корпоратив в Алматы 11 октября, ведущий, бюджет 1500000',
    rare:'Свадьба в Алматы 10 октября, флорист, бюджет 500000',
    absent:'Свадьба в Зарубежье 10 октября, флорист, бюджет 500000',
    empty:'Фотосессия на корпоратив, в Алмате на 25 октября, в 15:00 на 3 часа, бюджет 50тыс, язык не важен'
  };
  document.querySelectorAll('[data-preset]').forEach(button => button.addEventListener('click', () => { reset(); send(demos[button.dataset.preset]); }));
  refresh().then(reset).catch(error => add('bot', escapeHtml(error.message)));
})();
