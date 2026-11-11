import type {Metadata} from 'next';
import 'katex/dist/katex.min.css';
import './globals.css'; // Global styles

export const metadata: Metadata = {
  title: 'QuantMetrics: Stock Analysis & Volatility Modeling',
  description: 'Quantitative stock analysis terminal featuring ACF/PACF, GARCH and EWMA volatility modeling with forecast bands, seasonal decomposition, ADF stationarity tests, correlation heatmaps, and regression diagnostics.',
  openGraph: {
    title: 'QuantMetrics: Stock Analysis & Volatility Modeling',
    description: 'Quantitative stock analysis terminal featuring ACF/PACF, GARCH and EWMA volatility modeling with forecast bands, seasonal decomposition, ADF stationarity tests, correlation heatmaps, and regression diagnostics.',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'QuantMetrics: Stock Analysis & Volatility Modeling',
    description: 'Quantitative stock analysis terminal featuring ACF/PACF, GARCH and EWMA volatility modeling with forecast bands, seasonal decomposition, ADF stationarity tests, correlation heatmaps, and regression diagnostics.',
  },
};

export default function RootLayout({children}: {children: React.ReactNode}) {
  return (
    <html lang="en" className="dark">
      <body suppressHydrationWarning className="bg-[#faf8f5] dark:bg-[#121110] text-stone-900 dark:text-stone-100 min-h-screen antialiased selection:bg-amber-500/30 selection:text-amber-200">
        {children}
      </body>
    </html>
  );
}
