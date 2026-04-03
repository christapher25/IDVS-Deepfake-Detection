import { Heart, Home, MessageCircle, PlusSquare, Search, Compass } from "lucide-react";

interface InstagramHeaderProps {
  onUploadClick: () => void;
}

export function InstagramHeader({ onUploadClick }: InstagramHeaderProps) {
  return (
    <header className="sticky top-0 z-50 bg-white border-b border-gray-300">
      <div className="max-w-[975px] mx-auto px-5 h-[60px] flex items-center justify-between">
        {/* Logo */}
        <div className="w-[103px]">
          <svg aria-label="Instagram" role="img" viewBox="0 0 24 24" className="h-[29px]">
            <title>Instagram</title>
            <path d="M12 2.982c2.937 0 3.285.011 4.445.064 1.072.049 1.655.228 2.042.379.513.2.88.437 1.265.822.385.385.622.752.822 1.265.151.387.33.97.379 2.042.053 1.16.064 1.508.064 4.445s-.011 3.285-.064 4.445c-.049 1.072-.228 1.655-.379 2.042-.2.513-.437.88-.822 1.265-.385.385-.752.622-1.265.822-.387.151-.97.33-2.042.379-1.16.053-1.508.064-4.445.064s-3.285-.011-4.445-.064c-1.072-.049-1.655-.228-2.042-.379-.513-.2-.88-.437-1.265-.822-.385-.385-.622-.752-.822-1.265-.151-.387-.33-.97-.379-2.042-.053-1.16-.064-1.508-.064-4.445s.011-3.285.064-4.445c.049-1.072.228-1.655.379-2.042.2-.513.437-.88.822-1.265.385-.385.752-.622 1.265-.822.387-.151.97-.33 2.042-.379 1.16-.053 1.508-.064 4.445-.064M12 1c-2.987 0-3.362.013-4.535.066-1.171.054-1.971.24-2.67.511-.724.282-1.338.659-1.948 1.269-.61.61-.987 1.224-1.269 1.948-.271.699-.457 1.499-.511 2.67C1.013 8.638 1 9.013 1 12s.013 3.362.066 4.535c.054 1.171.24 1.971.511 2.67.282.724.659 1.338 1.269 1.948.61.61 1.224.987 1.948 1.269.699.271 1.499.457 2.67.511C8.638 22.987 9.013 23 12 23s3.362-.013 4.535-.066c1.171-.054 1.971-.24 2.67-.511.724-.282 1.338-.659 1.948-1.269.61-.61.987-1.224 1.269-1.948.271-.699.457-1.499.511-2.67.053-1.173.066-1.548.066-4.535s-.013-3.362-.066-4.535c-.054-1.171-.24-1.971-.511-2.67-.282-.724-.659-1.338-1.269-1.948-.61-.61-1.224-.987-1.948-1.269-.699-.271-1.499-.457-2.67-.511C15.362 1.013 14.987 1 12 1z" fill="currentColor"></path>
            <path d="M12 6.351c-3.115 0-5.649 2.534-5.649 5.649s2.534 5.649 5.649 5.649 5.649-2.534 5.649-5.649-2.534-5.649-5.649-5.649zM12 15.338c-2.024 0-3.662-1.638-3.662-3.662S9.976 8.014 12 8.014s3.662 1.638 3.662 3.662-1.638 3.662-3.662 3.662z" fill="currentColor"></path>
            <circle cx="17.846" cy="6.154" r="1.32" fill="currentColor"></circle>
          </svg>
        </div>

        {/* Search Bar */}
        <div className="hidden md:flex items-center bg-[#efefef] rounded-lg px-4 h-9 w-[268px]">
          <Search className="w-4 h-4 text-[#8e8e8e] mr-3" />
          <input
            type="text"
            placeholder="Search"
            className="bg-transparent outline-none text-sm w-full placeholder:text-[#8e8e8e]"
          />
        </div>

        {/* Navigation */}
        <nav className="flex items-center gap-5">
          <Home className="w-6 h-6 cursor-pointer stroke-[2]" strokeWidth={2} />
          <MessageCircle className="w-6 h-6 cursor-pointer stroke-[2]" strokeWidth={2} />
          <PlusSquare
            className="w-6 h-6 cursor-pointer stroke-[2]"
            strokeWidth={2}
            onClick={onUploadClick}
          />
          <Compass className="w-6 h-6 cursor-pointer stroke-[2]" strokeWidth={2} />
          <Heart className="w-6 h-6 cursor-pointer stroke-[2]" strokeWidth={2} />
          <div className="w-[22px] h-[22px] rounded-full overflow-hidden border border-black cursor-pointer">
            <img
              src="https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?w=100"
              alt="Profile"
              className="w-full h-full object-cover"
            />
          </div>
        </nav>
      </div>
    </header>
  );
}
