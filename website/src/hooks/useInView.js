import { useEffect, useRef, useState } from 'react';

/**
 * Scroll-triggered visibility hook. Latches to true the first time the
 * element intersects the viewport.
 */
const useInView = ({ threshold = 0.15, root = null, rootMargin = '0px' } = {}) => {
  const ref = useRef(null);
  const [isInView, setIsInView] = useState(false);

  useEffect(() => {
    const node = ref.current;
    if (!node || isInView) return undefined;

    if (typeof IntersectionObserver === 'undefined') {
      setIsInView(true);
      return undefined;
    }

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsInView(true);
          observer.disconnect();
        }
      },
      { threshold, root, rootMargin }
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [threshold, root, rootMargin, isInView]);

  return [ref, isInView];
};

export default useInView;
