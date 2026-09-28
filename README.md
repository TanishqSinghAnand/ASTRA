# ASTRA

**On-board procedural assistant for experiments aboard the Bharatiya Antariksh Station**

Smart India Hackathon 2026 · Problem statement **SIH26174**, AI Human Activity Recognition for On-board BAS Experiments (ISRO) · Software

ASTRA watches an experiment through one fixed camera and checks every pick and place against the written procedure while it happens. If an action is not the step the procedure expects, the crew gets a specific message about three seconds later: what was expected, what was seen, and which step is current. When the mistake is corrected, ASTRA confirms the recovery and the run carries on.

Everything runs on the machine the camera is plugged into, on an ordinary CPU, with no network connection. A new experiment needs a new JSON procedure file, not new training data.

## Built with

- Python 3.10+, FastAPI, WebSockets
- OpenCV, MediaPipe Hands, Ultralytics YOLOv8 (pretrained)
- Next.js, React, TypeScript for the dashboard
- pyttsx3 for offline voice prompts
- pytest
