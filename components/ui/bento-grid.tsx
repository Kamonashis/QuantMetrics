import React from 'react';

/** Aceternity Bento Grid — skewed 12-col grid for the Home capability showcase. */
export function BentoGrid({
  className = '',
  children,
}: {
  className?: string;
  children?: React.ReactNode;
}) {
  return (
    <div
      className={`mx-auto grid max-w-7xl grid-cols-1 gap-4 md:grid-cols-3 md:auto-rows-[15rem] ${className}`}
    >
      {children}
    </div>
  );
}

export function BentoGridItem({
  className = '',
  title,
  description,
  header,
  icon,
  onClick,
}: {
  className?: string;
  title?: string | React.ReactNode;
  description?: string | React.ReactNode;
  header?: React.ReactNode;
  icon?: React.ReactNode;
  onClick?: () => void;
}) {
  const Wrapper = onClick ? 'button' : 'div';
  return (
    <Wrapper
      onClick={onClick}
      type={onClick ? 'button' : undefined}
      className={`group row-span-1 flex cursor-pointer flex-col justify-between space-y-3 rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-[#101014] p-5 transition duration-300 hover:-translate-y-0.5 hover:border-gold-500/40 dark:hover:border-gold-400/30 hover:shadow-[0_18px_44px_-18px_rgba(222,192,132,0.22)] focus:outline-none focus-visible:ring-2 focus-visible:ring-gold-400/60 text-left ${className}`}
    >
      {header && <div className="flex min-h-0 flex-1 items-end">{header}</div>}
      <div className="transition-colors duration-300">
        <div className="mb-2 text-gold-700 dark:text-gold-300">{icon}</div>
        <div className="mb-1 font-display text-[17px] font-semibold leading-snug text-zinc-900 dark:text-zinc-100">
          {title}
        </div>
        <div className="text-[12.5px] leading-relaxed text-zinc-500 dark:text-zinc-400">{description}</div>
      </div>
    </Wrapper>
  );
}
