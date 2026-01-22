const path = require('path')

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  webpack: (config, { buildId, dev, isServer, defaultLoaders, webpack }) => {
    // Fix for plotly.js memory issues
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
      }
    }
    // Explicitly set path aliases for webpack - must match tsconfig.json
    config.resolve.alias = {
      ...(config.resolve.alias || {}),
      '@': path.resolve(__dirname),
      '@/lib': path.resolve(__dirname, 'lib'),
      '@/components': path.resolve(__dirname, 'components'),
      '@/app': path.resolve(__dirname, 'app'),
    }
    return config
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        // In production, proxy API requests to the FastAPI backend on Railway
        destination: 'https://6d-production.up.railway.app/api/:path*',
      },
    ]
  },
  // Increase timeout for long-running requests
  serverRuntimeConfig: {
    // This doesn't directly affect rewrites, but we'll handle it in the API route if needed
  },
}

module.exports = nextConfig

