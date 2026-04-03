import { Heart, MessageCircle, Send, Bookmark, MoreHorizontal, AlertTriangle } from "lucide-react";
import { Post } from "../../../App"; // Import the Post interface

interface InstagramFeedProps {
  posts: Post[];
}

export function InstagramFeed({ posts }: InstagramFeedProps) {
  return (
    <div className="space-y-6 pb-12">
      {posts.map((post) => (
        <article key={post.id} className="bg-white border border-gray-300 rounded-sm relative group">
          {/* Header */}
          <div className="flex items-center justify-between px-4 py-3">
            <div className="flex items-center gap-3">
              <img src={post.userAvatar} alt={post.username} className="w-8 h-8 rounded-full object-cover" />
              <div>
                <p className="font-semibold text-sm">{post.username}</p>
                {post.location && <p className="text-xs">{post.location}</p>}
              </div>
            </div>
            <MoreHorizontal className="w-5 h-5 cursor-pointer" />
          </div>

          {/* Media Content */}
          <div className="relative w-full bg-black">
            {post.mediaType === "video" ? (
              <video src={post.mediaUrl} controls className="w-full max-h-[600px] object-contain" />
            ) : (
              <img src={post.mediaUrl} alt="Post" className="w-full object-cover" />
            )}

            {/* --- DEEPFAKE WARNING LABEL --- */}
            {post.isDeepfake && (
              <div className="absolute bottom-0 left-0 w-full bg-red-600/90 backdrop-blur-md text-white px-4 py-3 flex items-center justify-between z-10">
                <div className="flex items-center gap-3">
                  <div className="bg-white/20 p-2 rounded-full">
                    <AlertTriangle className="w-5 h-5 text-white" />
                  </div>
                  <div className="flex flex-col">
                    <span className="text-xs font-bold uppercase tracking-wider text-white">AI Manipulated Media</span>
                    <span className="text-[11px] text-white/90">{post.deepfakeReason || "Digital artifacts detected"}</span>
                  </div>
                </div>
                <button className="text-[10px] font-bold border border-white/40 px-3 py-1.5 rounded hover:bg-white/10 transition">
                  Learn More
                </button>
              </div>
            )}
          </div>

          {/* Actions & Caption */}
          <div className="px-4 py-3">
            <div className="flex justify-between mb-2">
              <div className="flex gap-4">
                <Heart className="w-6 h-6 hover:opacity-50 cursor-pointer" />
                <MessageCircle className="w-6 h-6 hover:opacity-50 cursor-pointer" />
                <Send className="w-6 h-6 hover:opacity-50 cursor-pointer" />
              </div>
              <Bookmark className="w-6 h-6 hover:opacity-50 cursor-pointer" />
            </div>

            {post.likes > 0 && <p className="font-semibold text-sm mb-1">{post.likes.toLocaleString()} likes</p>}

            <div className="text-sm">
              <span className="font-semibold mr-2">{post.username}</span>
              {post.caption}
            </div>
            <p className="text-[10px] text-gray-500 uppercase mt-2">{post.timeAgo}</p>
          </div>
        </article>
      ))}
    </div>
  );
}