// Fundamental factor construction from normalized Yahoo quoteSummary statements.
// Pure TS. Reporting lag is enforced here: a statement's fiscal period end is
// NOT a filing date, so the value is only usable after `periodEnd + lagDays`.

import type { FactorDefinition } from './types';

export interface FundamentalStatement {
  /** Fiscal period end, YYYY-MM-DD. Yahoo does not expose the filing date. */
  periodEnd: string;
  values: Record<string, number | null>;
}

export interface FundamentalSnapshot {
  symbol: string;
  /** First date this snapshot was observed by the app (YYYY-MM-DD). */
  observedAt: string;
  sector: string | null;
  industry: string | null;
  marketCap: number | null;
  sharesOutstanding: number | null;
  price: number | null;
  statements: {
    annualIncome: FundamentalStatement[];
    quarterlyIncome: FundamentalStatement[];
    annualBalance: FundamentalStatement[];
    quarterlyBalance: FundamentalStatement[];
    annualCashflow: FundamentalStatement[];
    quarterlyCashflow: FundamentalStatement[];
  };
}

export interface FundamentalFactorOptions {
  /** Calendar days before a fiscal period end becomes usable. */
  quarterlyLagDays?: number;
  annualLagDays?: number;
  /** Optional price override (current market cap is a snapshot, not point-in-time). */
  marketCapOverride?: number | null;
}

const FIELD_ALIASES: Record<string, string[]> = {
  totalRevenue: ['totalRevenue', 'revenue'],
  costOfRevenue: ['costOfRevenue', 'costOfGoodsAndServicesSold'],
  grossProfit: ['grossProfit'],
  operatingIncome: ['operatingIncome', 'operatingIncomeLoss'],
  ebit: ['ebit', 'operatingIncome', 'incomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest'],
  ebitda: ['ebitda'],
  netIncome: ['netIncome', 'netIncomeCommonStockholders', 'netIncomeLoss'],
  pretaxIncome: ['incomeTaxExpenseBenefit', 'incomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest'],
  taxExpense: ['incomeTaxExpenseBenefit'],
  totalAssets: ['totalAssets'],
  bookEquity: ['stockholdersEquity', 'totalStockholderEquity', 'stockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'],
  totalDebt: ['totalDebt', 'longTermDebt', 'longTermDebtNoncurrent'],
  cash: ['cashAndCashEquivalentsAtCarryingValue', 'cashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents', 'cash'],
  currentAssets: ['assetsCurrent', 'totalCurrentAssets'],
  currentLiabilities: ['liabilitiesCurrent', 'totalCurrentLiabilities'],
  operatingCashFlow: ['operatingCashFlow', 'netCashProvidedByUsedInOperatingActivities'],
  capitalExpenditure: ['capitalExpenditure', 'paymentsToAcquirePropertyPlantAndEquipment'],
  dividendsPaid: ['paymentsOfDividends', 'paymentsOfDividendsCommonStock'],
  buybacks: ['paymentsForRepurchaseOfCommonStock', 'repurchaseOfCapitalStock'],
  issuances: ['proceedsFromIssuanceOfCommonStock', 'issuanceOfCommonStock'],
  interestExpense: ['interestExpense', 'interestExpenseNonOperating'],
  dilutedShares: ['dilutedAverageShares', 'weightedAverageNumberOfDilutedSharesOutstanding'],
};

function pick(row: FundamentalStatement | null, field: keyof typeof FIELD_ALIASES): number | null {
  if (!row) return null;
  for (const alias of FIELD_ALIASES[field]) {
    const raw = row.values[alias];
    if (typeof raw === 'number' && Number.isFinite(raw)) return raw;
  }
  return null;
}

function dateValue(ymd: string): number {
  const t = Date.parse(`${ymd}T00:00:00Z`);
  return Number.isFinite(t) ? t : NaN;
}

function availableAt(statement: FundamentalStatement, lagDays: number): string | null {
  const t = dateValue(statement.periodEnd);
  if (!Number.isFinite(t)) return null;
  return new Date(t + lagDays * 86_400_000).toISOString().slice(0, 10);
}

function pickUsable(
  statements: readonly FundamentalStatement[],
  asOfYmd: string,
  lagDays: number
): FundamentalStatement | null {
  let best: FundamentalStatement | null = null;
  let bestTime = -Infinity;
  for (const st of statements) {
    const available = availableAt(st, lagDays);
    if (!available || available > asOfYmd) continue;
    const t = dateValue(st.periodEnd);
    if (Number.isFinite(t) && t > bestTime) {
      best = st;
      bestTime = t;
    }
  }
  return best;
}

function ratio(numerator: number | null, denominator: number | null): number | null {
  if (numerator === null || denominator === null || Math.abs(denominator) < 1e-9) return null;
  const v = numerator / denominator;
  return Number.isFinite(v) ? v : null;
}

function growth(current: number | null, previous: number | null): number | null {
  if (current === null || previous === null || Math.abs(previous) < 1e-9) return null;
  const v = (current - previous) / Math.abs(previous);
  return Number.isFinite(v) ? v : null;
}

export const FUNDAMENTAL_FACTORS: readonly FactorDefinition[] = [
  { id: 'fund:EARNINGS_YIELD', family: 'value', label: 'Earnings Yield', formula: 'Net Income / Market Cap', rationale: 'Yield avoids the discontinuity of raw P/E when earnings cross zero.', direction: 1, requires: 'fundamentals', minObservations: 1 },
  { id: 'fund:BOOK_TO_MARKET', family: 'value', label: 'Book-to-Market', formula: 'Stockholders Equity / Market Cap', rationale: 'Classic HML value signal expressed as a yield.', direction: 1, requires: 'fundamentals', minObservations: 1 },
  { id: 'fund:EBITDA_EV', family: 'value', label: 'EBITDA / Enterprise Value', formula: 'EBITDA / (Market Cap + Total Debt − Cash)', rationale: 'Capital-structure-neutral operating value.', direction: 1, requires: 'fundamentals', minObservations: 1 },
  { id: 'fund:EBIT_EV', family: 'value', label: 'EBIT / Enterprise Value', formula: 'EBIT / (Market Cap + Total Debt − Cash)', rationale: 'Operating-profit value less sensitive to D&A policy.', direction: 1, requires: 'fundamentals', minObservations: 1 },
  { id: 'fund:FCF_YIELD', family: 'value', label: 'Free Cash Flow Yield', formula: '(Operating Cash Flow − CapEx) / Market Cap', rationale: 'Harder to manipulate than GAAP net income.', direction: 1, requires: 'fundamentals', minObservations: 1 },
  { id: 'fund:SHAREHOLDER_YIELD', family: 'value', label: 'Shareholder Yield', formula: '(Dividends + Buybacks − Issuances) / Market Cap', rationale: 'Total cash returned to shareholders.', direction: 1, requires: 'fundamentals', minObservations: 1 },
  { id: 'fund:GROSS_PROFITABILITY', family: 'quality', label: 'Gross Profitability (Novy-Marx)', formula: '(Revenue − Cost of Revenue) / Total Assets', rationale: 'One of the cleanest cross-sectional quality predictors.', direction: 1, requires: 'fundamentals', minObservations: 1 },
  { id: 'fund:OPERATING_PROFITABILITY', family: 'quality', label: 'Operating Profitability (RMW)', formula: 'Operating Income / Book Equity', rationale: 'Fama–French profitability proxy.', direction: 1, requires: 'fundamentals', minObservations: 1 },
  { id: 'fund:ROE', family: 'quality', label: 'Return on Equity', formula: 'Net Income / Stockholders Equity', rationale: 'Shareholder-level profitability.', direction: 1, requires: 'fundamentals', minObservations: 1 },
  { id: 'fund:ROIC', family: 'quality', label: 'Return on Invested Capital', formula: 'EBIT·(1−tax) / (Total Debt + Equity − Cash)', rationale: 'Capital efficiency independent of financing mix.', direction: 1, requires: 'fundamentals', minObservations: 1 },
  { id: 'fund:ACCRUALS', family: 'quality', label: 'Accruals Quality', formula: '−(Net Income − Operating Cash Flow) / Total Assets', rationale: 'Favors earnings backed by cash generation.', direction: 1, requires: 'fundamentals', minObservations: 1 },
  { id: 'fund:DEBT_TO_EQUITY', family: 'quality', label: 'Debt-to-Equity', formula: 'Total Debt / Book Equity', rationale: 'Balance-sheet leverage.', direction: -1, requires: 'fundamentals', minObservations: 1 },
  { id: 'fund:NET_DEBT_EBITDA', family: 'quality', label: 'Net Debt / EBITDA', formula: '(Total Debt − Cash) / EBITDA', rationale: 'Solvency and leverage capacity.', direction: -1, requires: 'fundamentals', minObservations: 1 },
  { id: 'fund:CURRENT_RATIO', family: 'quality', label: 'Current Ratio', formula: 'Current Assets / Current Liabilities', rationale: 'Short-term liquidity.', direction: 1, requires: 'fundamentals', minObservations: 1 },
  { id: 'fund:INTEREST_COVERAGE', family: 'quality', label: 'Interest Coverage', formula: 'EBIT / Interest Expense', rationale: 'Ability to service debt.', direction: 1, requires: 'fundamentals', minObservations: 1 },
  { id: 'fund:PIOTROSKI_F', family: 'quality', label: 'Piotroski F-Score', formula: 'Sum of 9 binary accounting signals', rationale: 'Discipline against financially distressed firms.', direction: 1, requires: 'fundamentals', minObservations: 2 },
  { id: 'fund:ASSET_GROWTH', family: 'investment', label: 'Asset Growth (CMA)', formula: '−(Total Assets(t) − Total Assets(t−1)) / Total Assets(t−1)', rationale: 'Aggressive balance-sheet expansion historically underperforms.', direction: 1, requires: 'fundamentals', minObservations: 2 },
  { id: 'fund:NET_SHARE_ISSUANCE', family: 'investment', label: 'Net Share Issuance', formula: '−Δ diluted shares / diluted shares', rationale: 'Captures dilution risk.', direction: 1, requires: 'fundamentals', minObservations: 2 },
  { id: 'fund:REV_CAGR_3Y', family: 'investment', label: '3-Year Revenue CAGR', formula: '(Revenue(t) / Revenue(t−3))^(1/3) − 1', rationale: 'Fundamental growth momentum.', direction: 1, requires: 'fundamentals', minObservations: 4 },
  { id: 'fund:EPS_CAGR_3Y', family: 'investment', label: '3-Year EPS CAGR', formula: '(Net Income(t) / Net Income(t−3))^(1/3) − 1', rationale: 'Earnings growth persistence.', direction: 1, requires: 'fundamentals', minObservations: 4 },
  { id: 'fund:MARGIN_EXPANSION', family: 'investment', label: 'Operating Margin Expansion', formula: 'Operating Margin(t) − Operating Margin(t−1)', rationale: 'Improving operating leverage.', direction: 1, requires: 'fundamentals', minObservations: 2 },
  { id: 'fund:SIZE_SMB', family: 'size', label: 'Log Market Cap (SMB)', formula: '−ln(Price × Shares Outstanding)', rationale: 'Log scale prevents mega-cap outliers dominating cross-sectional z-scores.', direction: 1, requires: 'fundamentals', minObservations: 1 },
] as const;

function computePiotroski(
  annualIncome: readonly FundamentalStatement[],
  annualBalance: readonly FundamentalStatement[],
  annualCashflow: readonly FundamentalStatement[]
): number | null {
  if (annualIncome.length < 2 || annualBalance.length < 2 || annualCashflow.length < 2) return null;
  const y = annualIncome[0];
  const yPrev = annualIncome[1];
  const b = annualBalance[0];
  const bPrev = annualBalance[1];
  const c = annualCashflow[0];
  const ni = pick(y, 'netIncome');
  const assets = pick(b, 'totalAssets');
  const assetsPrev = pick(bPrev, 'totalAssets');
  const equity = pick(b, 'bookEquity');
  const equityPrev = pick(bPrev, 'bookEquity');
  const cfo = pick(c, 'operatingCashFlow');
  const roa = ratio(ni, assets);
  const roaPrev = ratio(pick(yPrev, 'netIncome'), assetsPrev);
  const leverage = ratio(pick(b, 'totalDebt'), assets);
  const leveragePrev = ratio(pick(bPrev, 'totalDebt'), assetsPrev);
  const current = ratio(pick(b, 'currentAssets'), pick(b, 'currentLiabilities'));
  const currentPrev = ratio(pick(bPrev, 'currentAssets'), pick(bPrev, 'currentLiabilities'));
  const gm = ratio(pick(y, 'grossProfit'), pick(y, 'totalRevenue'));
  const gmPrev = ratio(pick(yPrev, 'grossProfit'), pick(yPrev, 'totalRevenue'));
  const turnover = ratio(pick(y, 'totalRevenue'), assets);
  const turnoverPrev = ratio(pick(yPrev, 'totalRevenue'), assetsPrev);
  const shares = pick(y, 'dilutedShares');
  const sharesPrev = pick(yPrev, 'dilutedShares');
  if (
    roa === null || roaPrev === null || leverage === null || leveragePrev === null ||
    current === null || currentPrev === null || gm === null || gmPrev === null ||
    turnover === null || turnoverPrev === null || shares === null || sharesPrev === null ||
    cfo === null || equity === null || equityPrev === null || assets === null
  ) return null;

  let score = 0;
  if (roa > 0) score++;
  if (cfo > 0) score++;
  if (roa > roaPrev) score++;
  if (cfo / assets > roa) score++;
  if (leverage < leveragePrev) score++;
  if (current > currentPrev) score++;
  if (shares <= sharesPrev) score++;
  if (gm > gmPrev) score++;
  if (turnover > turnoverPrev) score++;
  return score;
}

/** Compute all fundamental factors for one symbol as of `asOfYmd`. */
export function computeFundamentalFactors(
  snapshot: FundamentalSnapshot | null | undefined,
  asOfYmd: string,
  options: FundamentalFactorOptions = {}
): Record<string, number | null> {
  const ids = FUNDAMENTAL_FACTORS.map((f) => f.id);
  const out: Record<string, number | null> = {};
  for (const id of ids) out[id] = null;
  if (!snapshot) return out;

  const quarterlyLag = options.quarterlyLagDays ?? 90;
  const annualLag = options.annualLagDays ?? 180;
  const s = snapshot.statements;
  const income = pickUsable(s.quarterlyIncome ?? [], asOfYmd, quarterlyLag);
  const annualIncome = pickUsable(s.annualIncome ?? [], asOfYmd, annualLag);
  const balance = pickUsable(s.quarterlyBalance ?? [], asOfYmd, quarterlyLag);
  const annualBalance = pickUsable(s.annualBalance ?? [], asOfYmd, annualLag);
  const cashflow = pickUsable(s.quarterlyCashflow ?? [], asOfYmd, quarterlyLag);
  const annualCashflow = pickUsable(s.annualCashflow ?? [], asOfYmd, annualLag);
  const marketCap =
    options.marketCapOverride ?? (Number.isFinite(snapshot.marketCap) ? snapshot.marketCap : null);
  const shares = Number.isFinite(snapshot.sharesOutstanding) ? snapshot.sharesOutstanding : null;

  if (marketCap === null && shares === null) return out;

  const revenue = pick(income ?? annualIncome, 'totalRevenue');
  const cost = pick(income ?? annualIncome, 'costOfRevenue');
  const grossProfit = pick(income ?? annualIncome, 'grossProfit');
  const operatingIncome = pick(income ?? annualIncome, 'operatingIncome');
  const ebit = pick(income ?? annualIncome, 'ebit');
  const ebitda = pick(income ?? annualIncome, 'ebitda');
  const netIncome = pick(income ?? annualIncome, 'netIncome');
  const tax = pick(income ?? annualIncome, 'taxExpense');
  const pretax = pick(income ?? annualIncome, 'pretaxIncome');
  const dilutedShares = pick(income ?? annualIncome, 'dilutedShares') ?? shares;

  const assets = pick(balance ?? annualBalance, 'totalAssets');
  const equity = pick(balance ?? annualBalance, 'bookEquity');
  const debt = pick(balance ?? annualBalance, 'totalDebt');
  const cash = pick(balance ?? annualBalance, 'cash');
  const currentAssets = pick(balance ?? annualBalance, 'currentAssets');
  const currentLiabilities = pick(balance ?? annualBalance, 'currentLiabilities');

  const cfo = pick(cashflow ?? annualCashflow, 'operatingCashFlow');
  const capex = pick(cashflow ?? annualCashflow, 'capitalExpenditure');
  const dividends = pick(cashflow ?? annualCashflow, 'dividendsPaid');
  const buybacks = pick(cashflow ?? annualCashflow, 'buybacks');
  const issuances = pick(cashflow ?? annualCashflow, 'issuances');

  if (marketCap !== null) {
    out['fund:EARNINGS_YIELD'] = ratio(netIncome, marketCap);
    out['fund:BOOK_TO_MARKET'] = ratio(equity, marketCap);
    out['fund:SHAREHOLDER_YIELD'] = ratio(
      dividends !== null && buybacks !== null && issuances !== null
        ? dividends + buybacks - issuances
        : null,
      marketCap
    );
    if (cfo !== null && capex !== null) out['fund:FCF_YIELD'] = ratio(cfo - capex, marketCap);
    const impliedCap = marketCap ?? (shares !== null && snapshot.price !== null ? shares * snapshot.price : null);
    if (impliedCap !== null && impliedCap > 0) out['fund:SIZE_SMB'] = -Math.log(impliedCap);
  }

  const enterpriseValue =
    marketCap !== null && debt !== null && cash !== null ? marketCap + debt - cash : null;
  out['fund:EBITDA_EV'] = ratio(ebitda, enterpriseValue);
  out['fund:EBIT_EV'] = ratio(ebit, enterpriseValue);

  out['fund:GROSS_PROFITABILITY'] = ratio(
    grossProfit !== null ? grossProfit : revenue !== null && cost !== null ? revenue - cost : null,
    assets
  );
  out['fund:OPERATING_PROFITABILITY'] = ratio(operatingIncome, equity);
  out['fund:ROE'] = ratio(netIncome, equity);
  const taxRate = pretax !== null && tax !== null && pretax !== 0 ? tax / pretax : 0.21;
  out['fund:ROIC'] = ratio(
    ebit !== null ? ebit * (1 - Math.min(0.5, Math.max(0, taxRate))) : null,
    debt !== null && equity !== null && cash !== null ? debt + equity - cash : null
  );
  out['fund:ACCRUALS'] = ratio(netIncome !== null && cfo !== null ? cfo - netIncome : null, assets);
  out['fund:DEBT_TO_EQUITY'] = ratio(debt, equity);
  out['fund:NET_DEBT_EBITDA'] = ratio(
    debt !== null && cash !== null ? debt - cash : null,
    ebitda
  );
  out['fund:CURRENT_RATIO'] = ratio(currentAssets, currentLiabilities);
  out['fund:INTEREST_COVERAGE'] = ratio(ebit, pick(income ?? annualIncome, 'interestExpense'));

  const annualIncomeSorted = [...(s.annualIncome ?? [])].sort((a, b) =>
    b.periodEnd.localeCompare(a.periodEnd)
  );
  const annualBalanceSorted = [...(s.annualBalance ?? [])].sort((a, b) =>
    b.periodEnd.localeCompare(a.periodEnd)
  );
  const annualCashflowSorted = [...(s.annualCashflow ?? [])].sort((a, b) =>
    b.periodEnd.localeCompare(a.periodEnd)
  );
  out['fund:PIOTROSKI_F'] = computePiotroski(annualIncomeSorted, annualBalanceSorted, annualCashflowSorted);

  const assetsPrev = annualBalanceSorted[1] ? pick(annualBalanceSorted[1], 'totalAssets') : null;
  out['fund:ASSET_GROWTH'] = growth(assets, assetsPrev) === null ? null : -growth(assets, assetsPrev)!;

  const sharesPrev = annualIncomeSorted[1] ? pick(annualIncomeSorted[1], 'dilutedShares') ?? shares : null;
  const issuanceGrowth = growth(dilutedShares, sharesPrev);
  out['fund:NET_SHARE_ISSUANCE'] = issuanceGrowth === null ? null : -issuanceGrowth;

  if (annualIncomeSorted.length >= 4) {
    const r0 = pick(annualIncomeSorted[0], 'totalRevenue');
    const r3 = pick(annualIncomeSorted[3], 'totalRevenue');
    if (r0 !== null && r3 !== null && r3 > 0 && r0 > 0) {
      out['fund:REV_CAGR_3Y'] = Math.pow(r0 / r3, 1 / 3) - 1;
    }
    const n0 = pick(annualIncomeSorted[0], 'netIncome');
    const n3 = pick(annualIncomeSorted[3], 'netIncome');
    if (n0 !== null && n3 !== null && n3 > 0 && n0 > 0) {
      out['fund:EPS_CAGR_3Y'] = Math.pow(n0 / n3, 1 / 3) - 1;
    }
  }
  const opMargin = ratio(operatingIncome, revenue);
  const opMarginPrev = annualIncomeSorted[1]
    ? ratio(
        pick(annualIncomeSorted[1], 'operatingIncome'),
        pick(annualIncomeSorted[1], 'totalRevenue')
      )
    : null;
  if (opMargin !== null && opMarginPrev !== null) {
    out['fund:MARGIN_EXPANSION'] = opMargin - opMarginPrev;
  }

  for (const id of ids) {
    const v = out[id];
    if (v !== null && !Number.isFinite(v)) out[id] = null;
  }
  return out;
}
