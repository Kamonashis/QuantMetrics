/**
 * One chart palette: gold is always the primary series, zinc the secondary,
 * desaturated green/red are semantic up/down. No sky/indigo/purple data series.
 */
export const CHART_PALETTE = {
  gold: '#DEC084',
  goldDim: 'rgba(222,192,132,0.45)',
  zinc: '#a1a1aa',
  zincDim: 'rgba(161,161,170,0.45)',
  positive: '#3fb27f',
  negative: '#f43f5e',
  axis: '#8b8b93',
  grid: 'rgba(161,161,170,0.14)',
} as const;

/** Stable series order — same colour means the same role in every chart. */
export const SERIES_COLORS: string[] = [
  CHART_PALETTE.gold,
  CHART_PALETTE.zinc,
  CHART_PALETTE.positive,
  CHART_PALETTE.negative,
  '#c9a227',
];
