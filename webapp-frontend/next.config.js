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
    return config
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/api/:path*',
      },
    ]
  },
  // Increase timeout for long-running requests
  serverRuntimeConfig: {
    // This doesn't directly affect rewrites, but we'll handle it in the API route if needed
  },
}

module.exports = nextConfig

