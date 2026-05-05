import { motion } from "framer-motion";

export default function MetricCard({ label, value, accent = "from-violet-500 to-blue-500" }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className="rounded-2xl border border-white/20 bg-white/30 p-4 shadow-md backdrop-blur-xl dark:border-white/10 dark:bg-white/5"
    >
      <span className="text-xs uppercase tracking-wide text-slate-600 dark:text-slate-300">{label}</span>
      <div className={`mt-3 h-1.5 w-full rounded-full bg-gradient-to-r ${accent}`} />
      <p className="mt-3 text-2xl font-semibold">{value}</p>
    </motion.div>
  );
}
