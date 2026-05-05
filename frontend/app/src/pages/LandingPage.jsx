import { motion } from "framer-motion";
import { ArrowRight, Sparkles } from "lucide-react";
import { Link } from "react-router-dom";

export default function LandingPage() {
  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-slate-950 px-4">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_20%_25%,rgba(139,92,246,0.3),transparent_35%),radial-gradient(circle_at_80%_15%,rgba(56,189,248,0.25),transparent_35%),radial-gradient(circle_at_55%_80%,rgba(45,212,191,0.25),transparent_45%)]" />
      <motion.div
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, ease: "easeOut" }}
        className="relative mx-auto w-full max-w-3xl rounded-[2rem] border border-white/20 bg-white/10 p-6 text-center shadow-2xl backdrop-blur-2xl md:p-12"
      >
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="mx-auto mb-6 flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-violet-500 to-cyan-500"
        >
          <Sparkles />
        </motion.div>
        <h1 className="bg-gradient-to-r from-violet-200 via-cyan-100 to-teal-100 bg-clip-text text-4xl font-semibold text-transparent md:text-6xl">
          AI Fruit Analysis
        </h1>
        <p className="mx-auto mt-5 max-w-xl text-sm text-slate-200 md:text-lg">
          Analyze fruit quality using machine learning
        </p>
        <Link
          to="/analyze"
          className="mt-8 inline-flex items-center gap-2 rounded-2xl bg-gradient-to-r from-violet-500 to-blue-500 px-6 py-3 text-sm font-semibold text-white shadow-lg transition hover:scale-[1.03] hover:shadow-violet-500/30 md:text-base"
        >
          Start Analysis <ArrowRight size={18} />
        </Link>
      </motion.div>
    </div>
  );
}
