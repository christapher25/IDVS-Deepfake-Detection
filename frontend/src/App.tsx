import { useState } from "react";
import { InstagramHeader } from "./app/components/figma/InstagramHeader";
import { InstagramStories } from "./app/components/figma/InstagramStories";
import { InstagramFeed } from "./app/components/figma/InstagramFeed";
import { InstagramSidebar } from "./app/components/figma/InstagramSidebar";
import { VideoUploadModal } from "./app/components/figma/VideoUploadModal";

// 1. Define what a "Post" looks like
export interface Post {
  id: number;
  username: string;
  userAvatar: string;
  location: string;
  mediaUrl: string; // URL for the image or video
  mediaType: "image" | "video";
  likes: number;
  caption: string;
  comments: number;
  timeAgo: string;
  isDeepfake: boolean; // <--- The critical flag
  deepfakeReason?: string;
}

export default function App() {
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);

  // 2. State to hold all posts (Starting with one dummy post)
  const [posts, setPosts] = useState<Post[]>([
    {
      id: 1,
      username: "mountain_explorer",
      userAvatar: "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=100",
      location: "Swiss Alps, Switzerland",
      mediaUrl: "https://images.unsplash.com/photo-1616386573884-22531fd226e6?w=1080",
      mediaType: "image",
      likes: 12847,
      caption: "The mountains are calling! 🏔️",
      comments: 234,
      timeAgo: "2 HOURS AGO",
      isDeepfake: false,
    },
  ]);

  // 3. Function to add a new post to the top of the list
  const handleNewPost = (newPost: Post) => {
    setPosts((prevPosts) => [newPost, ...prevPosts]);
  };

  return (
    <div className="min-h-screen bg-[#fafafa]">
      <InstagramHeader onUploadClick={() => setIsUploadModalOpen(true)} />

      <div className="max-w-[935px] mx-auto pt-[30px] px-5 flex gap-8">
        <main className="flex-1 max-w-[614px]">
          <InstagramStories />
          {/* Pass the posts state to the Feed */}
          <InstagramFeed posts={posts} />
        </main>

        <InstagramSidebar />
      </div>

      <VideoUploadModal
        isOpen={isUploadModalOpen}
        onClose={() => setIsUploadModalOpen(false)}
        onPostCreate={handleNewPost} // Pass the function to the modal
      />
    </div>
  );
}