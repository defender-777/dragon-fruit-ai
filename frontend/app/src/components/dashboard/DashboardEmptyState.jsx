import { motion } from "framer-motion";
import { Sparkles } from "lucide-react";
import { Link } from "react-router-dom";

export default function DashboardEmptyState() {
  return (
    <motion.section
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="flex flex-col items-center justify-center rounded-3xl border border-white/20 bg-white/30 p-12 text-center shadow-xl backdrop-blur-2xl dark:border-white/10 dark:bg-white/5"
    >
      <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-violet-500 to-cyan-500 shadow-lg shadow-violet-500/20">
        <Sparkles className="text-white" size={32} />
      </div>
      <h1 className="text-2xl font-semibold md:text-3xl">Ready for Analysis</h1>
      <p className="mt-4 max-w-sm text-slate-600 dark:text-slate-300">
        You haven't performed any fruit scans yet. Head over to the analysis page to get started.
      </p>
      <Link
        to="/analyze"
        className="mt-8 rounded-2xl bg-gradient-to-r from-violet-600 to-blue-600 px-8 py-3 text-sm font-semibold text-white shadow-lg transition hover:scale-105 hover:shadow-violet-500/30"
      >
        Go to Analyze
      </Link>
    </motion.section>
  );
}
