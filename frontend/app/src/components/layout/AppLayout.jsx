import { AnimatePresence, motion } from "framer-motion";
import { Apple, History, LayoutDashboard, Moon, Sparkles, Sun } from "lucide-react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { useDarkMode } from "../../utils/useDarkMode";

const navItems = [
  { to: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { to: "/analyze", label: "Analyze", icon: Sparkles },
  { to: "/history", label: "History", icon: History },
];

function NavLinks() {
  return (
    <ul className="space-y-2">
      {navItems.map(({ to, label, icon: Icon }) => (
        <li key={to}>
          <NavLink
            to={to}
            className={({ isActive }) =>
              `flex items-center gap-3 rounded-2xl border px-4 py-3 transition-all duration-200 ${
                isActive
                  ? "border-white/30 bg-white/20 text-white shadow-lg"
                  : "border-transparent bg-white/5 text-slate-200 hover:border-white/20 hover:bg-white/15"
              }`
            }
          >
            <Icon size={18} />
            <span className="text-sm font-medium">{label}</span>
          </NavLink>
        </li>
      ))}
    </ul>
  );
}

export default function AppLayout() {
  const { isDark, toggle } = useDarkMode();
  const location = useLocation();

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 transition-colors dark:bg-slate-950 dark:text-white">
      <div className="pointer-events-none fixed inset-0 bg-[radial-gradient(circle_at_10%_20%,rgba(79,70,229,0.25),transparent_30%),radial-gradient(circle_at_90%_10%,rgba(45,212,191,0.25),transparent_30%),radial-gradient(circle_at_60%_90%,rgba(168,85,247,0.2),transparent_40%)]" />
      <div className="relative mx-auto flex min-h-screen max-w-7xl flex-col gap-4 px-4 py-4 md:px-6 md:py-6 lg:flex-row">
        <aside className="w-full rounded-3xl border border-white/20 bg-white/40 p-4 shadow-xl backdrop-blur-2xl dark:border-white/10 dark:bg-white/5 lg:w-72">
          <div className="mb-6 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="rounded-2xl bg-gradient-to-br from-violet-500 to-cyan-500 p-2.5 shadow-lg">
                <Apple size={18} />
              </div>
              <div>
                <p className="text-sm font-semibold tracking-wide">FruitScope</p>
                <p className="text-xs text-slate-600 dark:text-slate-300">AI Analysis App</p>
              </div>
            </div>
            <button
              type="button"
              onClick={toggle}
              className="rounded-xl border border-white/20 bg-white/20 p-2 transition hover:scale-105 dark:border-white/10 dark:bg-white/10"
              aria-label="Toggle color mode"
            >
              {isDark ? <Sun size={16} /> : <Moon size={16} />}
            </button>
          </div>
          <NavLinks />
        </aside>

        <main className="w-full rounded-3xl border border-white/20 bg-white/40 p-4 shadow-xl backdrop-blur-2xl dark:border-white/10 dark:bg-white/5 md:p-6">
          <AnimatePresence mode="wait">
            <motion.div
              key={location.pathname}
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              transition={{ duration: 0.3, ease: "easeOut" }}
            >
              <Outlet />
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}
