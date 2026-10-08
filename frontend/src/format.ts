export const percent = (n: number) =>
  new Intl.NumberFormat('fr-FR', { style: 'percent', maximumFractionDigits: 1 }).format(n);
export const count = (n: number) => new Intl.NumberFormat('fr-FR').format(n);
export const score = (n: number) =>
  n !== 0 && Math.abs(n) < 0.0001
    ? n.toExponential(4)
    : n.toLocaleString('fr-FR', { maximumSignificantDigits: 6 });
