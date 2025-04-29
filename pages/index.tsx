import { Canvas } from '@react-three/fiber';
import { useRef, useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as posedetection from '@tensorflow-models/pose-detection';
import { Line } from '@react-three/drei'; 

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [keypoints, setKeypoints] = useState<posedetection.Keypoint[]>([]);

  useEffect(() => {
    const runPoseDetection = async () => {
      await tf.setBackend('webgl');
      const detector = await posedetection.createDetector(posedetection.SupportedModels.MoveNet);

      if (navigator.mediaDevices.getUserMedia) {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }

        const detectPose = async () => {
          if (videoRef.current) {
            const poses = await detector.estimatePoses(videoRef.current);
            if (poses.length > 0) {
              setKeypoints(poses[0].keypoints);
            }
          }
          requestAnimationFrame(detectPose);
        };
        detectPose();
      }
    };

    runPoseDetection();
  }, []);

  return (
    <div className="w-screen h-screen overflow-hidden bg-black">
      <video ref={videoRef} className="hidden" playsInline muted />
      <Canvas camera={{ position: [0, 0, 2.5] }}>
        <ambientLight />
        <Skeleton keypoints={keypoints} />
      </Canvas>
    </div>
  );
}

const Skeleton = ({ keypoints }: { keypoints: posedetection.Keypoint[] }) => {
  if (keypoints.length === 0) return null;

  const adjacentPairs = posedetection.util.getAdjacentPairs(posedetection.SupportedModels.MoveNet);

  return (
    <>
      {adjacentPairs.map(([i, j], idx) => {
        const kp1 = keypoints[i];
        const kp2 = keypoints[j];
        if (kp1.score > 0.5 && kp2.score > 0.5) {
          const points = [
            [normalize(kp1.x), normalize(-kp1.y), getDepth(kp1)],
            [normalize(kp2.x), normalize(-kp2.y), getDepth(kp2)],
          ];
          return <Line key={idx} points={points} color="cyan" lineWidth={3} />;
        }
      })}
    </>
  );
};

const normalize = (value: number) => (value / 640) * 2 - 1;
const getDepth = (keypoint: posedetection.Keypoint) => -0.5;
