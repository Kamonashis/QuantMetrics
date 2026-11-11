import type {Metadata, Viewport} from 'next';
import {Space_Grotesk, Inter, IBM_Plex_Mono} from 'next/font/google';
import 'katex/dist/katex.min.css';
import './globals.css'; // Global styles

const spaceGrotesk = Space_Grotesk({
  subsets: ['latin'],
  variable: '--font-space-grotesk',
  display: 'swap',
});

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
});

const plexMono = IBM_Plex_Mono({
  subsets: ['latin'],
  weight: ['400', '500', '600'],
  variable: '--font-plex-mono',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'QuantMetrics: Stock Analysis & Volatility Modeling',
  description:
    'Quantitative stock analysis terminal featuring ACF/PACF, GARCH and EWMA volatility modeling with forecast bands, seasonal decomposition, ADF stationarity tests, correlation heatmaps, and regression diagnostics.',
  applicationName: 'QuantMetrics',
  openGraph: {
    title: 'QuantMetrics: Stock Analysis & Volatility Modeling',
    description:
      'Quantitative stock analysis terminal featuring ACF/PACF, GARCH and EWMA volatility modeling with forecast bands, seasonal decomposition, ADF stationarity tests, correlation heatmaps, and regression diagnostics.',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'QuantMetrics: Stock Analysis & Volatility Modeling',
    description:
      'Quantitative stock analysis terminal featuring ACF/PACF, GARCH and EWMA volatility modeling with forecast bands, seasonal decomposition, ADF stationarity tests, correlation heatmaps, and regression diagnostics.',
  },
};

export const viewport: Viewport = {
  themeColor: [
    {media: '(prefers-color-scheme: dark)', color: '#08080A'},
    {media: '(prefers-color-scheme: light)', color: '#F5F4F1'},
  ],
};

export default function RootLayout({children}: {children: React.ReactNode}) {
  return (
    <html lang="en" className={`dark ${spaceGrotesk.variable} ${inter.variable} ${plexMono.variable}`}>
      <body suppressHydrationWarning className="bg-[#F5F4F1] dark:bg-[#08080A] text-zinc-900 dark:text-[#ECECF1] font-sans min-h-screen antialiased">
        {children}
      </body>
    </html>
  );
}
