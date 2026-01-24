/**
 * Next.js Instrumentation
 * 
 * This file runs before the server starts, allowing us to configure
 * Node.js settings globally before any requests are handled.
 * 
 * Fixes ECONNREFUSED errors in Docker by forcing IPv4 DNS resolution.
 */

export async function register() {
    if (process.env.NEXT_RUNTIME === 'nodejs') {
        const dns = await import('node:dns');
        dns.setDefaultResultOrder('ipv4first');
        console.log('[instrumentation] DNS resolution order set to ipv4first');
    }
}
