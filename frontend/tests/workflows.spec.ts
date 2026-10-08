import { expect, test } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import fs from 'node:fs/promises';
test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await expect(page.getByText('Service prêt', { exact: true })).toBeVisible();
});
test('real API example, chunked inference, review, dialog, export and navigation', async ({
  page,
}) => {
  const requests: number[] = [];
  page.on('request', (request) => {
    if (request.url().endsWith('/v1/predict-batch'))
      requests.push(JSON.parse(request.postData()!).rows.length);
  });
  await page.getByRole('button', { name: 'Essayer un exemple', exact: true }).click();
  await page.getByRole('button', { name: 'Analyser les mesures' }).click();
  await expect(page.getByRole('heading', { name: 'Des scores à examiner.' })).toBeVisible();
  expect(requests).toEqual([2, 1]);
  await page.getByRole('checkbox', { name: 'Mesure 1 examinée' }).check();
  await page.getByRole('button', { name: 'Voir mesure 1', exact: true }).click();
  await expect(page.getByRole('dialog')).toBeVisible();
  await page.getByLabel('Rechercher un capteur').fill('sensor_0');
  await expect(page.getByRole('dialog').getByText('sensor_0', { exact: true })).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(page.getByRole('dialog')).not.toBeVisible();
  const downloadEvent = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Exporter la revue' }).click();
  const download = await downloadEvent;
  expect(download.suggestedFilename()).toBe('fleetguard-review.json');
  const saved = JSON.parse(await fs.readFile((await download.path())!, 'utf8'));
  expect(saved.rows).toBe(3);
  expect(saved.model.dataset_kind).toBe('synthetic_smoke');
  expect(saved.predictions[0].reviewed).toBe(true);
  await page.getByRole('button', { name: /Diagnostic du modèle/ }).click();
  await expect(page.getByRole('heading', { name: 'Ce que la décision manque.' })).toBeVisible();
  await expect(page.getByText('Non défini · aucun positif', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: /Revue des alertes/ }).click();
  await expect(page.getByRole('checkbox', { name: 'Mesure 1 examinée' })).toBeChecked();
});
test('CSV upload restores sensor schema and rejects labels before any inference', async ({
  page,
}) => {
  const info = await (await page.request.get('/v1/model')).json();
  const header = info.feature_names.slice().reverse();
  await page.getByLabel('Importer un CSV').setInputFiles({
    name: 'sensors.csv',
    mimeType: 'text/csv',
    buffer: Buffer.from(header.join(',') + '\n' + header.map(() => 'na').join(',')),
  });
  await expect(page.getByText('sensors.csv', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Analyser les mesures' }).click();
  await expect(page.getByRole('heading', { name: 'Des scores à examiner.' })).toBeVisible();
  await page
    .getByLabel('Importer un CSV')
    .setInputFiles({ name: 'labels.csv', mimeType: 'text/csv', buffer: Buffer.from('class\npos') });
  await expect(page.getByRole('alert')).toContainText('Colonnes incompatibles');
  await expect(page.getByRole('button', { name: 'Analyser les mesures' })).toBeDisabled();
  await expect(page.getByRole('heading', { name: 'Des scores à examiner.' })).not.toBeVisible();
});
test('error or identity drift in a later batch never exposes partial predictions', async ({
  page,
}) => {
  let calls = 0;
  await page.route('**/v1/predict-batch', async (route) => {
    const response = await route.fetch();
    const data = await response.json();
    if (++calls === 2) data.model.release_id = 'replaced-model';
    await route.fulfill({ response, json: data });
  });
  await page.getByRole('button', { name: 'Essayer un exemple', exact: true }).click();
  await page.getByRole('button', { name: 'Analyser les mesures' }).click();
  await expect(page.getByRole('alert')).toContainText('modèle a changé');
  await expect(page.getByRole('heading', { name: 'Des scores à examiner.' })).not.toBeVisible();
});
test('cancellation clears progress outcome and leaves the input available', async ({ page }) => {
  await page.route('**/v1/predict-batch', async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 500));
    await route.abort();
  });
  await page.getByRole('button', { name: 'Essayer un exemple', exact: true }).click();
  await page.getByRole('button', { name: 'Analyser les mesures' }).click();
  await page.getByRole('button', { name: 'Annuler', exact: true }).click();
  await expect(page.getByRole('status')).toContainText('Analyse annulée');
  await expect(page.getByRole('button', { name: 'Analyser les mesures' })).toBeEnabled();
  await expect(page.getByRole('heading', { name: 'Des scores à examiner.' })).not.toBeVisible();
});
test('offline API can reconnect and reference results remain available', async ({ page }) => {
  await page.route('**/health/ready', (route) => route.fulfill({ status: 503, body: 'not ready' }));
  await page.reload();
  await expect(page.getByRole('alert')).toContainText('HTTP 503');
  await expect(page.getByRole('button', { name: 'Analyser les mesures' })).toBeDisabled();
  await page.getByRole('button', { name: /Expériences/ }).click();
  await expect(
    page.getByRole('heading', { name: 'Comparer dans un même protocole.' }),
  ).toBeVisible();
  await page.getByLabel('Étape').selectOption('comparison');
  await expect(page.getByRole('cell', { name: 'random_forest', exact: true })).toBeVisible();
  await page.unroute('**/health/ready');
  await page.getByRole('button', { name: /Revue des alertes/ }).click();
  await page.getByRole('button', { name: 'Reconnecter le service', exact: true }).last().click();
  await expect(page.getByText('Service prêt', { exact: true })).toBeVisible();
});
test('keyboard and automated accessibility checks cover the result and native dialog', async ({
  page,
}) => {
  await page.getByRole('button', { name: 'Essayer un exemple', exact: true }).click();
  await page.getByRole('button', { name: 'Analyser les mesures' }).click();
  await expect(page.getByRole('heading', { name: 'Des scores à examiner.' })).toBeVisible();
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
  await page.getByRole('button', { name: 'Voir mesure 1', exact: true }).focus();
  await page.keyboard.press('Enter');
  await expect(page.getByRole('button', { name: 'Fermer', exact: true })).toBeFocused();
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
  await page.keyboard.press('Escape');
  await expect(page.getByRole('button', { name: 'Voir mesure 1', exact: true })).toBeFocused();
});
test('mobile layout remains usable without page-wide horizontal overflow', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole('button', { name: 'Essayer un exemple', exact: true }).click();
  await page.getByRole('button', { name: 'Analyser les mesures' }).click();
  await expect(page.getByRole('heading', { name: 'Des scores à examiner.' })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(
    true,
  );
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
  await page.getByRole('button', { name: /Diagnostic du modèle/ }).click();
  await expect(
    page.getByRole('heading', { name: 'Chaque tranche a son dénominateur.' }),
  ).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(
    true,
  );
});
