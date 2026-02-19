"use client";

import { useState } from "react";
import axios from "axios";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Loader2, Film, Star, ExternalLink, AlertCircle } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface Recommendation {
  title: string;
  year: string;
  score: number;
  cf_score: number;
  vibe_similarity: number;
  why: string;
  overview?: string;
  poster_url?: string;
}

export default function Home() {
  const [url, setUrl] = useState("");
  const [count, setCount] = useState(10);
  const [loading, setLoading] = useState(false);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [error, setError] = useState<string | null>(null);

  const getRecommendations = async () => {
    if (!url) return;
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post("http://localhost:8000/recommend", {
        url: url,
        count: count,
        debug: true,
      });
      setRecommendations(response.data.recommendations);
    } catch (err: any) {
      setError(err.response?.data?.detail || "Something went wrong. Please check the URL and try again.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen p-8 md:p-16 max-w-7xl mx-auto">
      <header className="mb-12 text-center md:text-left">
        <div className="inline-block bg-accent neobrutalism-border neobrutalism-shadow px-4 py-1 mb-4">
          <h2 className="text-sm font-black uppercase tracking-widest">AI Movie Scout</h2>
        </div>
        <h1 className="text-5xl md:text-7xl font-black mb-4 uppercase leading-tight">
          Vibe-Matched <br />
          <span className="text-main">Recommendations</span>
        </h1>
        <p className="text-xl font-bold max-w-2xl text-gray-800">
          Paste your Letterboxd profile link. We&apos;ll analyze your taste profile and find movies that match your unique cinematic &quot;vibe&quot;.
        </p>
      </header>

      <div className="flex flex-col md:flex-row gap-4 mb-16 max-w-3xl">
        <Input
          placeholder="https://letterboxd.com/username/"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          className="text-lg flex-[3]"
          onKeyDown={(e) => e.key === "Enter" && getRecommendations()}
        />
        <div className="flex items-center gap-2 flex-1">
          <span className="font-bold text-sm uppercase opacity-60 whitespace-nowrap">Count:</span>
          <Input
            type="number"
            min={1}
            max={50}
            value={count}
            onChange={(e) => setCount(parseInt(e.target.value) || 1)}
            className="text-lg w-20 text-center"
          />
        </div>
        <Button
          onClick={getRecommendations}
          disabled={loading}
          size="lg"
          className="whitespace-nowrap"
        >
          {loading ? <Loader2 className="mr-2 h-5 w-5 animate-spin" /> : "FIND MY VIBE"}
        </Button>
      </div>

      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="mb-8 p-4 bg-red-100 neobrutalism-border border-red-500 text-red-700 font-bold flex items-center gap-3"
          >
            <AlertCircle /> {error}
          </motion.div>
        )}
      </AnimatePresence>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
        {recommendations.map((rec, idx) => (
          <motion.div
            key={rec.title + idx}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.1 }}
          >
            <Card className="h-full flex flex-col group">
              <div className="relative aspect-[2/3] overflow-hidden neobrutalism-border bg-gray-200 mb-4">
                {rec.poster_url ? (
                  <img
                    src={rec.poster_url}
                    alt={rec.title}
                    className="object-cover w-full h-full"
                  />
                ) : (
                  <div className="w-full h-full flex flex-col items-center justify-center text-gray-400">
                    <Film size={48} />
                    <p className="mt-2 font-bold tracking-tighter">NO POSTER</p>
                  </div>
                )}
                <div className="absolute top-4 right-4 bg-accent neobrutalism-border neobrutalism-shadow-sm px-2 py-1 flex items-center gap-1 font-black transform transition-transform group-hover:rotate-3">
                  <Star size={16} fill="black" />
                  {rec.vibe_similarity.toFixed(2)}
                </div>
              </div>

              <CardHeader>
                <CardTitle className="group-hover:text-main transition-colors uppercase">
                  {rec.title} {rec.year && <span className="text-gray-500 text-lg">({rec.year})</span>}
                </CardTitle>
                <CardDescription className="line-clamp-3 font-bold text-gray-700">
                  {rec.overview || "No overview available for this film."}
                </CardDescription>
              </CardHeader>

              <CardContent className="mt-4 flex-grow flex flex-col">
                <div className="mt-auto p-3 bg-secondary/20 neobrutalism-border border-dashed text-xs font-bold leading-relaxed">
                  <span className="uppercase text-[10px] block opacity-60 mb-1">Why this?</span>
                  {rec.why}
                </div>
              </CardContent>

              <CardFooter className="bg-gray-50 -mx-6 -mb-6 p-4 mt-auto">
                <Button variant="outline" size="sm" className="w-full" asChild>
                  <a
                    href={`https://www.themoviedb.org/search?query=${encodeURIComponent(rec.title)}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center justify-center gap-2"
                  >
                    VIEW ON TMDB <ExternalLink size={14} />
                  </a>
                </Button>
              </CardFooter>
            </Card>
          </motion.div>
        ))}
      </div>

      {recommendations.length === 0 && !loading && !error && (
        <div className="text-center py-20 border-4 border-dashed border-black/10 rounded-3xl">
          <Film size={64} className="mx-auto mb-6 opacity-20" />
          <h3 className="text-2xl font-black text-black/20 uppercase tracking-widest">
            Enter your Letterboxd URL to see results
          </h3>
        </div>
      )}
    </main>
  );
}
