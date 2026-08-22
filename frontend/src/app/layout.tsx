import type { Metadata } from "next";
import { Geist_Mono, Inter, Space_Grotesk } from "next/font/google";
import "./globals.css";
import Sidebar from "@/components/sidebar";

const inter = Inter({ variable: "--font-inter", subsets: ["latin"] });
const spaceGrotesk = Space_Grotesk({ variable: "--font-space-grotesk", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "RAG Console",
  description: "Production RAG chatbot with hybrid retrieval and citations",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${inter.variable} ${spaceGrotesk.variable} ${geistMono.variable} antialiased`}
      >
        <div className="flex h-dvh">
          <Sidebar />
          <main className="min-w-0 flex-1 overflow-y-auto">{children}</main>
        </div>
      </body>
    </html>
  );
}
