const path = require('path')

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  webpack: (config, { isServer }) => {
    // Fix for plotly.js memory issues
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
      }
    }
    // Ensure path aliases work correctly
    config.resolve.alias = {
      ...config.resolve.alias,
      '@': path.resolve(__dirname),
    }
    return config
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        // In production, proxy API requests to the FastAPI backend on Vercel
        destination: 'https://6d-nu.vercel.app/api/:path*',
      },
    ]
  },
  // Increase timeout for long-running requests
  serverRuntimeConfig: {
    // This doesn't directly affect rewrites, but we'll handle it in the API route if needed
  },
}

module.exports = nextConfig

