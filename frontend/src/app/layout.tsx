import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Clipper MVP | AI Shorts Generator",
  description:
    "Transform YouTube streams and videos into high-retention vertical shorts automatically.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${inter.className} bg-zinc-950 text-zinc-100 antialiased min-h-screen selection:bg-amber-500 selection:text-black`}
      >
        {children}
      </body>
    </html>
  );
}
