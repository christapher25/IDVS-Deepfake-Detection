export function InstagramStories() {
  const stories = [
    { id: 1, username: "Your story", avatar: "https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?w=100", isYourStory: true },
    { id: 2, username: "nature_lover", avatar: "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=100", hasStory: true },
    { id: 3, username: "wildphotography", avatar: "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=100", hasStory: true },
    { id: 4, username: "earthpix", avatar: "https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=100", hasStory: true },
    { id: 5, username: "landscapes", avatar: "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=100", hasStory: true },
    { id: 6, username: "adventures", avatar: "https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=100", hasStory: true },
    { id: 7, username: "outdoor_life", avatar: "https://images.unsplash.com/photo-1554151228-14d9def656e4?w=100", hasStory: true },
    { id: 8, username: "travelblog", avatar: "https://images.unsplash.com/photo-1489424731084-a5d8b219a5bb?w=100", hasStory: true },
  ];

  return (
    <div className="bg-white border border-gray-300 rounded-lg p-4 mb-6">
      <div className="flex gap-4 overflow-x-auto scrollbar-hide">
        {stories.map((story) => (
          <div key={story.id} className="flex flex-col items-center gap-1 flex-shrink-0 cursor-pointer">
            <div className={`${story.isYourStory ? 'bg-gray-200' : 'bg-gradient-to-tr from-yellow-400 via-red-500 to-purple-600'} p-[2px] rounded-full`}>
              <div className="bg-white p-[2px] rounded-full">
                <div className="w-14 h-14 rounded-full overflow-hidden">
                  <img
                    src={story.avatar}
                    alt={story.username}
                    className="w-full h-full object-cover"
                  />
                </div>
              </div>
            </div>
            <span className="text-xs max-w-[66px] truncate">{story.username}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
