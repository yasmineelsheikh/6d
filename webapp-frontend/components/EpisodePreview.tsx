'use client'

interface EpisodePreviewProps {
  datasetData: any[]
}

export default function EpisodePreview({ datasetData }: EpisodePreviewProps) {
  // Show first 5 episodes
  const episodes = datasetData.slice(0, 5)

  if (episodes.length === 0) {
    return (
      <div>
        <h2 className="text-xs font-medium mb-3 text-[#d4d4d4]">Episode Preview</h2>
        <p className="text-[#8a8a8a] text-xs">No episodes available</p>
      </div>
    )
  }

  return (
    <div>
      <h2 className="text-xs font-medium mb-3 text-[#d4d4d4]">Episode Preview</h2>
      <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-2">
        {episodes.map((episode, idx) => (
          <div
            key={idx}
            className="bg-[#1a1a1a] border border-[#2a2a2a] p-2"
          >
            <div className="aspect-video bg-[#222222] mb-2 border border-[#2a2a2a] overflow-hidden">
              {episode.video_url ? (
                <video
                  src={episode.video_url}
                  className="w-full h-full object-cover"
                  controls
                  preload="metadata"
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center">
                  <span className="text-[#666666] text-xs font-mono">No video</span>
                </div>
              )}
            </div>
            <div className="text-xs font-medium text-[#d4d4d4] mb-1">
              {episode.id || `Episode ${idx + 1}`}
            </div>
            {episode.task_language_instruction && (
              <div className="text-xs text-[#8a8a8a] line-clamp-2">
                {episode.task_language_instruction}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

