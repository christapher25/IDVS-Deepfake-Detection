export function InstagramSidebar() {
  const suggestions = [
    { id: 1, username: "wildernessculture", name: "Wilderness Culture", avatar: "https://images.unsplash.com/photo-1554151228-14d9def656e4?w=100" },
    { id: 2, username: "beautifuldestinations", name: "Beautiful Destinations", avatar: "https://images.unsplash.com/photo-1508214751196-bcfd4ca60f91?w=100" },
    { id: 3, username: "natgeotravel", name: "National Geographic Travel", avatar: "https://images.unsplash.com/photo-1529626455594-4ff0802cfb7e?w=100" },
    { id: 4, username: "earthofficial", name: "Earth Official", avatar: "https://images.unsplash.com/photo-1524504388940-b1c1722653e1?w=100" },
    { id: 5, username: "wonderfulplaces", name: "Wonderful Places", avatar: "https://images.unsplash.com/photo-1504199367641-aba8151af406?w=100" },
  ];

  return (
    <aside className="hidden xl:block w-[320px] fixed right-0 top-[84px] px-8">
      {/* User Profile */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-4">
          <div className="w-14 h-14 rounded-full overflow-hidden">
            <img
              src="https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?w=100"
              alt="Your profile"
              className="w-full h-full object-cover"
            />
          </div>
          <div>
            <p className="font-semibold text-sm">your_username</p>
            <p className="text-sm text-gray-500">Your Name</p>
          </div>
        </div>
        <button className="text-xs text-[#0095f6] font-semibold">Switch</button>
      </div>

      {/* Suggestions Section */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <p className="text-sm text-gray-500 font-semibold">Suggestions For You</p>
          <button className="text-xs font-semibold">See All</button>
        </div>

        <div className="space-y-4">
          {suggestions.map((user) => (
            <div key={user.id} className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-full overflow-hidden">
                  <img
                    src={user.avatar}
                    alt={user.username}
                    className="w-full h-full object-cover"
                  />
                </div>
                <div>
                  <p className="font-semibold text-sm">{user.username}</p>
                  <p className="text-xs text-gray-500">Followed by user + 2 more</p>
                </div>
              </div>
              <button className="text-xs text-[#0095f6] font-semibold">Follow</button>
            </div>
          ))}
        </div>
      </div>

      {/* Footer */}
      <footer className="text-xs text-gray-400 space-y-3">
        <div className="flex flex-wrap gap-x-2 gap-y-1">
          <a href="#" className="hover:underline">About</a>
          <span>·</span>
          <a href="#" className="hover:underline">Help</a>
          <span>·</span>
          <a href="#" className="hover:underline">Press</a>
          <span>·</span>
          <a href="#" className="hover:underline">API</a>
          <span>·</span>
          <a href="#" className="hover:underline">Jobs</a>
          <span>·</span>
          <a href="#" className="hover:underline">Privacy</a>
          <span>·</span>
          <a href="#" className="hover:underline">Terms</a>
          <span>·</span>
          <a href="#" className="hover:underline">Locations</a>
          <span>·</span>
          <a href="#" className="hover:underline">Language</a>
        </div>
        <p className="text-xs">© 2026 INSTAGRAM FROM META</p>
      </footer>
    </aside>
  );
}
