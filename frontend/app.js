'use strict';
(() => {
  const el = id => document.getElementById(id);
  const escapeHtml = value => String(value).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  const money = value => Number(value).toLocaleString('ru-RU') + ' ₸';
  const date = value => value.split('-').reverse().join('.');
  let slots = {}, pendingFields = [], serial = 0, busy = false, datasets = [], datasetId = 'default', lastRecommendation = null;

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

  function summary(order = slots) {
    return [order.category, order.city, order.event_type, order.event_date && date(order.event_date),
      order.budget_kzt && money(order.budget_kzt), order.duration_hours && `${order.duration_hours} ч`, order.language].filter(Boolean).join(' · ');
  }

  function timingHTML(started) {
    const elapsed = (performance.now() - started) / 1000;
    return `<div class="response-time${elapsed > 10 ? ' slow' : ''}">Ответ за ${elapsed.toLocaleString('ru-RU', {minimumFractionDigits:2, maximumFractionDigits:2})} с · ${elapsed > 10 ? 'дольше ориентира 10 с' : 'ориентир ≤ 10 с'}</div>`;
  }

  function comparisonHTML(comparison) {
    if (!comparison) return '';
    let html = `<section class="comparison"><h3>Изменилась только дата: ${escapeHtml(date(comparison.previous_date))} → ${escapeHtml(date(comparison.event_date))}</h3><p>${escapeHtml(comparison.message)}</p>`;
    if (comparison.changes.length) {
      html += '<ul>' + comparison.changes.map(change => `<li><strong>${escapeHtml(change.anon_name)}</strong>: ${escapeHtml(change.message)}</li>`).join('') + '</ul>';
    }
    return html + '</section>';
  }

  function resultHTML(result) {
    const titles = {matched:'Подобрали подрядчиков.',category_not_in_city:'Нет такой категории в городе.',no_matches:'Кандидаты есть, но ни один не подходит.'};
    const style = {matched:'matched',category_not_in_city:'warn',no_matches:'bad'}[result.status];
    let html = `<div class="outcome-banner ${style}"><b>${titles[result.status]}</b><br>${escapeHtml(result.message)}</div>`;
    html += comparisonHTML(result.date_comparison);
    if (result.cards.length) {
      html += `<p class="breakdown">Кандидатов в городе: ${result.candidate_count} · прошли условия: ${result.eligible_count} · показаны первые ${result.cards.length}.<br>Порядок: цена «от» по возрастанию; при равной цене — код профиля.</p>`;
    }
    for (const [i,p] of result.cards.entries()) {
      const tags = [p.synthetic ? 'синтетический профиль' : 'несинтетический профиль (по CSV)'];
      if (datasetId !== 'default') tags.push('из загруженного CSV');
      if (p.city_imputed) tags.push('город проставлен при подготовке');
      if (p.price_imputed) tags.push('цену «от» нужно уточнить');
      html += `<div class="card"><div class="card-top"><div><span class="card-rank">#${i+1}</span><span class="card-name">${escapeHtml(p.anon_name)}</span><span class="card-code">${escapeHtml(p.id)}</span></div><div class="card-price">от ${money(p.price_from_kzt)}</div></div><div class="card-cat">${escapeHtml(p.category)} — ${escapeHtml(p.city)}</div><div class="card-tags">${tags.map(t=>`<span class="tag">${escapeHtml(t)}</span>`).join('')}</div><div class="card-explain">${escapeHtml(p.explanation)}</div></div>`;
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
    serial++; slots = {}; pendingFields = []; lastRecommendation = null; setBusy(false);
    el('chat').replaceChildren(); chips([]);
    const selected = datasets.find(d => d.id === datasetId);
    add('bot', 'Опишите мероприятие: город, дата, тип, категория и бюджет; необязательно — язык и часы. Например: «фотосессия на корпоратив в Алмате 25 октября на 3 часа, бюджет 50тыс».');
    if (selected) {
      add('bot', `Каталог: ${escapeHtml(selected.name)}. Календарь: ${date(selected.calendar_start)} — ${date(selected.calendar_end)}. В каком городе ищем?`);
      chips(selected.cities);
    }
  }

  function setBusy(value) {
    busy = value;
    el('sendBtn').disabled = value; el('sendBtn').textContent = value ? 'Подбираю…' : 'Отправить';
    el('repeatBtn').disabled = value || !lastRecommendation;
    el('datasetSelect').disabled = value;
    document.querySelectorAll('[data-preset], #compareDatesBtn, #showExamplesBtn').forEach(button => { button.disabled = value; });
  }

  function remember(result, order = slots) {
    lastRecommendation = {
      order:Object.fromEntries(Object.entries(order).filter(([, value]) => value != null)),
      datasetId, ids:result.cards.map(card => card.id), status:result.status
    };
  }

  async function send(text) {
    if (!text.trim() || busy) return;
    if (/^(сброс|заново|новый поиск|начать заново)$/i.test(text.trim())) { reset(); return; }
    add('user', escapeHtml(text));
    const current = ++serial; setBusy(true); chips([]);
    const started = performance.now();
    try {
      const response = await api('/chat', {message:text, slots, pending_fields:pendingFields, dataset_id:datasetId});
      if (current !== serial) return;
      slots = response.slots;
      pendingFields = response.pending_fields || [];
      let html = `<div class="slots-note">Распознано: ${escapeHtml(summary()) || 'пока нет параметров'}</div>`;
      for (const warning of response.warnings) html += `<p>${escapeHtml(warning)}</p>`;
      if (response.result) { html += resultHTML(response.result); remember(response.result); }
      else { html += escapeHtml(response.prompt); chips(response.choices); }
      add('bot', html + timingHTML(started));
      return {serial:current, hasResult:!!response.result};
    } catch (error) {
      if (current === serial) add('bot', `<div class="outcome-banner bad">${escapeHtml(error.message)}</div>${timingHTML(started)}`);
    } finally {
      if (current === serial) setBusy(false);
    }
  }

  async function repeat() {
    if (busy || !lastRecommendation || lastRecommendation.datasetId !== datasetId) return;
    const previous = lastRecommendation;
    add('user', `Повторить с теми же параметрами: ${escapeHtml(summary(previous.order))}`);
    const current = ++serial; setBusy(true); chips([]);
    const started = performance.now();
    try {
      const result = await api('/recommendations', {...previous.order, dataset_id:previous.datasetId});
      if (current !== serial) return;
      const same = previous.status === result.status && JSON.stringify(previous.ids) === JSON.stringify(result.cards.map(card => card.id));
      const message = same
        ? (result.cards.length ? `Порядок совпал: все ${result.cards.length} карточки на тех же местах.` : 'Пустой результат повторился с теми же параметрами.')
        : 'Результат изменился: порядок карточек или исход отличаются от предыдущего запроса.';
      slots = {...previous.order}; pendingFields = []; remember(result);
      add('bot', `<div class="outcome-banner ${same ? 'matched' : 'warn'}">${escapeHtml(message)} Сравнение выполнено по кодам профилей; параметры и каталог те же.</div>${resultHTML(result)}${timingHTML(started)}`);
    } catch (error) {
      if (current === serial) add('bot', `<div class="outcome-banner bad">${escapeHtml(error.message)}</div>${timingHTML(started)}`);
    } finally {
      if (current === serial) setBusy(false);
    }
  }

  function showDataset() {
    const d = datasets.find(d => d.id === datasetId);
    if (!d) return;
    const status = {ready:'семантический индекс готов',lexical:'локальный поиск',indexing:'индекс строится',degraded:'API недоступен — локальный поиск'}[d.index_status];
    el('datasetInfo').textContent = `${d.profile_count} профилей, из них синтетических ${d.synthetic_count}. ${status}. ${d.index_message}`;
    el('presets').hidden = datasetId !== 'default';
    const counts = d.category_counts;
    if (counts) {
      el('demoDescription').textContent = `В исходном каталоге: ведущих — ${counts['Ведущий'] || 0}, фотографов — ${counts['Фотограф'] || 0}, банкетных залов — ${counts['Банкетный зал'] || 0}; флористов — ${counts['Флорист'] || 0}. Примеры показывают ранжирование осенью, редкую категорию и запрос без результата.`;
    }
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
  el('repeatBtn').addEventListener('click', repeat);
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
    rare:'Свадьба в Алматы 10 октября, флорист, бюджет 500000',
    absent:'Свадьба в Зарубежье 10 октября, флорист, бюджет 500000',
    empty:'Фотосессия на корпоратив, в Алмате на 25 октября, в 15:00 на 3 часа, бюджет 50тыс, язык не важен'
  };
  document.querySelectorAll('[data-preset]').forEach(button => button.addEventListener('click', () => { if (busy) return; reset(); send(demos[button.dataset.preset]); }));
  el('showExamplesBtn').addEventListener('click', async () => {
    if (busy) return;
    reset();
    add('bot', 'Покажем три отдельных запроса: ведущие на осеннюю дату, флорист и подбор с бюджетом 50 тысяч. Все результаты останутся в диалоге.');
    let expected = serial;
    for (const key of ['dense', 'rare', 'empty']) {
      if (expected !== serial) return;
      slots = {}; pendingFields = [];
      const response = await send(demos[key]);
      if (!response || !response.hasResult || response.serial !== serial) return;
      expected = response.serial;
    }
  });
  el('compareDatesBtn').addEventListener('click', async () => {
    if (busy) return;
    reset();
    add('bot', 'Сначала подберём ведущих на 10 октября, затем изменим только дату на 11 октября. Оба результата останутся в диалоге, а изменения занятости будут объяснены.');
    const first = await send(demos.dense);
    if (first && first.hasResult && first.serial === serial) await send('11 октября');
  });
  refresh().then(reset).catch(error => add('bot', escapeHtml(error.message)));
})();
