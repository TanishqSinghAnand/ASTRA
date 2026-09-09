"use client";

import { useEffect, useRef, useState } from "react";
import { WS_INGEST_URL } from "./api";

const UPLOAD_HZ = 10; // frames/sec sent to the backend — a live status
// view doesn't need more, and each frame costs a getUserMedia canvas
// draw + JPEG encode + WS send, all on the main thread.
const JPEG_QUALITY = 0.7;
const RECONNECT_DELAY_MS = 2000;

export type BrowserCameraStatus =
  | "idle" // not active (source isn't "browser", or experiment isn't running)
  | "requesting" // getUserMedia permission prompt in flight
  | "streaming" // camera acquired, frames going out over /ws/ingest
  | "denied" // user (or the browser) refused camera access
  | "unavailable"; // no camera device, or getUserMedia unsupported (non-HTTPS, old browser)

/**
 * The inverse of useLiveSocket: when the backend has no local camera of
 * its own (camera.source == "browser", set for a cloud deployment), this
 * captures the *viewer's* own webcam via getUserMedia and streams JPEG
 * frames up to /ws/ingest — the backend runs the exact same perception
 * pipeline against those as it would against a locally-attached camera.
 *
 * `active` gates the whole thing (pass `cameraSource === "browser" &&
 * status === "RUNNING"`) so a visitor is never prompted for camera
 * permission before they've actually pressed Start, and the stream/socket
 * both tear down the instant the experiment stops.
 */
export function useBrowserCameraUpload(active: boolean) {
  const [status, setStatus] = useState<BrowserCameraStatus>("idle");
  const videoRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    if (!active) {
      setStatus("idle");
      return;
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      setStatus("unavailable");
      return;
    }

    let cancelled = false;
    let stream: MediaStream | null = null;
    let socket: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let captureTimer: ReturnType<typeof setInterval> | null = null;
    const video = document.createElement("video");
    const canvas = document.createElement("canvas");
    videoRef.current = video;

    const connectSocket = () => {
      if (cancelled) return;
      socket = new WebSocket(WS_INGEST_URL);
      socket.binaryType = "arraybuffer";
      socket.onclose = () => {
        if (!cancelled) reconnectTimer = setTimeout(connectSocket, RECONNECT_DELAY_MS);
      };
      socket.onerror = () => socket?.close();
    };

    const captureFrame = () => {
      if (socket?.readyState !== WebSocket.OPEN) return;
      if (video.videoWidth === 0) return; // metadata not loaded yet
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(video, 0, 0);
      canvas.toBlob(
        (blob) => {
          if (blob && socket?.readyState === WebSocket.OPEN) {
            blob.arrayBuffer().then((buf) => {
              if (socket?.readyState === WebSocket.OPEN) socket.send(buf);
            });
          }
        },
        "image/jpeg",
        JPEG_QUALITY,
      );
    };

    setStatus("requesting");
    navigator.mediaDevices
      .getUserMedia({ video: { facingMode: "environment" }, audio: false })
      .then((s) => {
        if (cancelled) {
          s.getTracks().forEach((t) => t.stop());
          return;
        }
        stream = s;
        video.srcObject = s;
        video.muted = true;
        video.playsInline = true;
        video.play().catch(() => {});
        setStatus("streaming");
        connectSocket();
        captureTimer = setInterval(captureFrame, 1000 / UPLOAD_HZ);
      })
      .catch((err) => {
        if (cancelled) return;
        setStatus(err?.name === "NotAllowedError" ? "denied" : "unavailable");
      });

    return () => {
      cancelled = true;
      if (captureTimer) clearInterval(captureTimer);
      if (reconnectTimer) clearTimeout(reconnectTimer);
      socket?.close();
      stream?.getTracks().forEach((t) => t.stop());
      video.srcObject = null;
    };
  }, [active]);

  return { status };
}
