import { normCdf } from './options';

export function creditBridge(equity: number | null | undefined, equityVol: number | null | undefined, debt?: number | null, rate = 0.04) {
  if (equity == null || equityVol == null || !(equity > 0 && equityVol > 0) || !Number.isFinite(equity + equityVol + rate)) return null;
  const barrier = debt ?? equity * 0.3;
  if (!(barrier > 0) || !Number.isFinite(barrier)) return null;
  const assetValue = equity + barrier;
  const assetVol = equityVol * equity / assetValue;
  const dd = (Math.log(assetValue / barrier) + rate - 0.5 * assetVol ** 2) / assetVol;
  return Number.isFinite(dd) ? { assetValue, assetVol, dd, pd: normCdf(-dd), debt: barrier } : null;
}
