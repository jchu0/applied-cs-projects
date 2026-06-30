'use client';

import Link from 'next/link';
import { useState } from 'react';

export default function ContactPage() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    company: '',
    subject: '',
    message: '',
  });
  const [submitted, setSubmitted] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    // Simulate form submission
    await new Promise((resolve) => setTimeout(resolve, 1000));

    setSubmitted(true);
    setLoading(false);
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b">
        <div className="container mx-auto px-4 h-16 flex items-center justify-between">
          <Link href="/" className="font-bold text-xl">
            SaaS Platform
          </Link>
          <nav className="flex items-center gap-4">
            <Link href="/features" className="text-sm text-muted-foreground hover:text-foreground">
              Features
            </Link>
            <Link href="/pricing" className="text-sm text-muted-foreground hover:text-foreground">
              Pricing
            </Link>
            <Link href="/login" className="text-sm">
              Log in
            </Link>
            <Link
              href="/register"
              className="text-sm bg-primary text-primary-foreground px-4 py-2 rounded-md hover:bg-primary/90"
            >
              Get Started
            </Link>
          </nav>
        </div>
      </header>

      <main className="flex-1 py-16">
        <div className="container mx-auto px-4">
          <div className="max-w-5xl mx-auto">
            <div className="grid md:grid-cols-2 gap-16">
              {/* Contact Info */}
              <div>
                <h1 className="text-4xl font-bold tracking-tight mb-4">Get in touch</h1>
                <p className="text-lg text-muted-foreground mb-8">
                  Have questions about our platform? Want to discuss enterprise solutions?
                  We&apos;re here to help.
                </p>

                <div className="space-y-6">
                  <div>
                    <h3 className="font-semibold mb-2">Sales</h3>
                    <p className="text-sm text-muted-foreground">
                      Interested in our enterprise plans or need a custom solution?
                    </p>
                    <a href="mailto:sales@saasplatform.com" className="text-sm text-primary hover:underline">
                      sales@saasplatform.com
                    </a>
                  </div>

                  <div>
                    <h3 className="font-semibold mb-2">Support</h3>
                    <p className="text-sm text-muted-foreground">
                      Need help with your account or have technical questions?
                    </p>
                    <a href="mailto:support@saasplatform.com" className="text-sm text-primary hover:underline">
                      support@saasplatform.com
                    </a>
                  </div>

                  <div>
                    <h3 className="font-semibold mb-2">Press</h3>
                    <p className="text-sm text-muted-foreground">
                      For media inquiries and press resources.
                    </p>
                    <a href="mailto:press@saasplatform.com" className="text-sm text-primary hover:underline">
                      press@saasplatform.com
                    </a>
                  </div>
                </div>

                <div className="mt-8 pt-8 border-t">
                  <h3 className="font-semibold mb-4">Office</h3>
                  <address className="text-sm text-muted-foreground not-italic">
                    123 Tech Street<br />
                    San Francisco, CA 94102<br />
                    United States
                  </address>
                </div>
              </div>

              {/* Contact Form */}
              <div>
                {submitted ? (
                  <div className="border rounded-lg p-8 text-center">
                    <div className="w-16 h-16 rounded-full bg-green-100 flex items-center justify-center mx-auto mb-4">
                      <CheckIcon className="w-8 h-8 text-green-600" />
                    </div>
                    <h2 className="text-xl font-semibold mb-2">Message sent!</h2>
                    <p className="text-muted-foreground mb-6">
                      Thank you for reaching out. We&apos;ll get back to you within 24 hours.
                    </p>
                    <button
                      onClick={() => {
                        setSubmitted(false);
                        setFormData({ name: '', email: '', company: '', subject: '', message: '' });
                      }}
                      className="text-sm text-primary hover:underline"
                    >
                      Send another message
                    </button>
                  </div>
                ) : (
                  <form onSubmit={handleSubmit} className="border rounded-lg p-8">
                    <h2 className="text-xl font-semibold mb-6">Send us a message</h2>

                    <div className="space-y-4">
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label htmlFor="name" className="block text-sm font-medium mb-1">
                            Name
                          </label>
                          <input
                            type="text"
                            id="name"
                            name="name"
                            value={formData.name}
                            onChange={handleChange}
                            required
                            className="w-full border rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                          />
                        </div>
                        <div>
                          <label htmlFor="email" className="block text-sm font-medium mb-1">
                            Email
                          </label>
                          <input
                            type="email"
                            id="email"
                            name="email"
                            value={formData.email}
                            onChange={handleChange}
                            required
                            className="w-full border rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                          />
                        </div>
                      </div>

                      <div>
                        <label htmlFor="company" className="block text-sm font-medium mb-1">
                          Company (optional)
                        </label>
                        <input
                          type="text"
                          id="company"
                          name="company"
                          value={formData.company}
                          onChange={handleChange}
                          className="w-full border rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                        />
                      </div>

                      <div>
                        <label htmlFor="subject" className="block text-sm font-medium mb-1">
                          Subject
                        </label>
                        <select
                          id="subject"
                          name="subject"
                          value={formData.subject}
                          onChange={handleChange}
                          required
                          className="w-full border rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                        >
                          <option value="">Select a topic</option>
                          <option value="sales">Sales inquiry</option>
                          <option value="support">Technical support</option>
                          <option value="billing">Billing question</option>
                          <option value="partnership">Partnership opportunity</option>
                          <option value="other">Other</option>
                        </select>
                      </div>

                      <div>
                        <label htmlFor="message" className="block text-sm font-medium mb-1">
                          Message
                        </label>
                        <textarea
                          id="message"
                          name="message"
                          value={formData.message}
                          onChange={handleChange}
                          required
                          rows={5}
                          className="w-full border rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary resize-none"
                        />
                      </div>

                      <button
                        type="submit"
                        disabled={loading}
                        className="w-full bg-primary text-primary-foreground py-2 rounded-md font-medium hover:bg-primary/90 disabled:opacity-50"
                      >
                        {loading ? 'Sending...' : 'Send message'}
                      </button>
                    </div>
                  </form>
                )}
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t py-8">
        <div className="container mx-auto px-4 text-center text-sm text-muted-foreground">
          <p>&copy; 2024 SaaS Platform. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}

function CheckIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 20 20"
      fill="currentColor"
      className={className}
    >
      <path
        fillRule="evenodd"
        d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z"
        clipRule="evenodd"
      />
    </svg>
  );
}
