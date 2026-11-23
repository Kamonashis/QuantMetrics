import type {NextConfig} from 'next';

const nextConfig: NextConfig = {
  reactStrictMode: true,
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: false,
  },
  // Allow access to remote image placeholder.
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'picsum.photos',
        port: '',
        pathname: '/**', // This allows any path under the hostname
      },
    ],
  },
  output: 'standalone',
  transpilePackages: ['motion'],
  allowedDevOrigins: [
    'localhost',
    '127.0.0.1',
    '192.168.1.6',
    'ais-dev-e3pcmk43szkohkhmajbzon-33949032970.asia-east1.run.app',
    'ais-pre-e3pcmk43szkohkhmajbzon-33949032970.asia-east1.run.app',
    '*.run.app',
  ],
  webpack: (config, {dev}) => {
    // HMR is disabled in AI Studio via DISABLE_HMR env var.
    // Do not modify—file watching is disabled to prevent flickering during agent edits.
    if (dev && process.env.DISABLE_HMR === 'true') {
      config.watchOptions = {
        ignored: /.*/,
      };
    }
    // Skip AST re-parsing of pre-bundled minified plotly.js-dist-min to keep builds fast and memory-safe
    const plotlyNoParse = /plotly\.js-dist-min/;
    config.module = config.module || {};
    const existingNoParse = config.module.noParse;
    config.module.noParse = existingNoParse
      ? Array.isArray(existingNoParse)
        ? [...existingNoParse, plotlyNoParse]
        : [existingNoParse, plotlyNoParse]
      : plotlyNoParse;
    return config;
  },
};

export default nextConfig;
