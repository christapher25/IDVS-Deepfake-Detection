import React, { useState } from 'react';

export function ImageWithFallback(props: React.ImgHTMLAttributes<HTMLImageElement>) {
  const [error, setError] = useState(false);
  const { src, alt, ...rest } = props;

  return error ? (
    <div className="bg-gray-200 w-full h-full flex items-center justify-center text-gray-400 text-xs">
      Image Failed
    </div>
  ) : (
    <img src={src} alt={alt} onError={() => setError(true)} {...rest} />
  );
}