# Voice Assistant: Pi + Jetson

A low-latency voice assistant for Home Assistant, using a Raspberry Pi 5 for audio capture and an NVIDIA Jetson Orin Nano for ML processing.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Raspberry Pi 5 (192.168.1.203)                  │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────────────┐   │
│  │ Microphone  │───▶│ VAD + Audio  │───▶│ WebSocket Client         │   │
│  └─────────────┘    │ Capture      │    │ (streams to Jetson)      │   │
│                     └──────────────┘    └──────────────────────────┘   │
│                                                    │                    │
│                                                    ▼                    │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────────────┐   │
│  │ Speaker     │◀───│ Audio        │◀───│ Home Assistant           │   │
│  │ (feedback)  │    │ Feedback     │    │ Controller               │   │
│  └─────────────┘    └──────────────┘    └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ WebSocket (audio stream)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Jetson Orin Nano (192.168.1.117)                   │
│  ┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐   │
│  │ WebSocket       │───▶│ Faster-Whisper   │───▶│ Intent Parser    │   │
│  │ Server          │    │ (GPU accelerated)│    │ (regex + LLM)    │   │
│  └─────────────────┘    └──────────────────┘    └──────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Features

- **Low Latency**: Streaming audio with VAD, GPU-accelerated transcription
- **Room Control**: Turn on/off entire rooms with a single command
- **Brightness Control**: Adjust dimmable lights by percentage
- **Audio Feedback**: Audio cues for listening, success, and errors
- **Automatic Reconnection**: Resilient to network interruptions

## Supported Commands

- "Turn on the living room"
- "Turn off the bedroom lights"
- "Set the office floor lamp to 50 percent"
- "Dim the bedside lamp to 30"
- "Toggle the kitchen light"
- "Turn on the Noguchi" (uses friendly names)

## Setup

### Prerequisites

#### Raspberry Pi 5
- Python 3.11+
- PortAudio library (for PyAudio)
- Home Assistant with long-lived access token

#### Jetson Orin Nano
- Python 3.10+
- CUDA 11.x or 12.x
- cuDNN 8.x

### Installation

#### On Raspberry Pi 5:

```bash
# Install system dependencies
sudo apt update
sudo apt install -y portaudio19-dev python3-pyaudio

# Clone the repository
git clone <repo-url> voice-assistant
cd voice-assistant

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements-pi.txt
```

#### On Jetson Orin Nano:

```bash
# Clone the repository
git clone <repo-url> voice-assistant
cd voice-assistant

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements-jetson.txt

# Copy the entities file from the Pi
scp bernardoct@192.168.1.203:~/.cache/ha_entities.json ./
```

### Configuration

#### Home Assistant Token

Generate a long-lived access token in Home Assistant:
1. Go to your Profile (click your username)
2. Scroll to "Long-Lived Access Tokens"
3. Create a new token and save it

#### Environment Variables

Create a `.env` file or export these variables:

```bash
# Raspberry Pi
export HA_TOKEN="your_home_assistant_token"
export HA_URL="http://192.168.1.203:8123"
export HA_ENTITIES_PATH="/home/bernardoct/.cache/ha_entities.json"
export JETSON_HOST="192.168.1.117"
export JETSON_PORT="8765"

# Jetson
export WHISPER_MODEL="base.en"  # Options: tiny.en, base.en, small.en
export WHISPER_DEVICE="cuda"
export WHISPER_COMPUTE_TYPE="float16"
```

### Running

#### 1. Start the Jetson Server

```bash
# On Jetson Orin Nano
cd voice-assistant
source venv/bin/activate
python run_jetson.py --entities ha_entities.json
```

Options:
- `--host`: Bind address (default: 0.0.0.0)
- `--port`: Port number (default: 8765)
- `--model`: Whisper model size (tiny.en, base.en, small.en)
- `--entities`: Path to ha_entities.json

#### 2. Start the Pi Client

```bash
# On Raspberry Pi
cd voice-assistant
source venv/bin/activate
python run_pi.py --ha-token "YOUR_TOKEN"
```

Options:
- `--jetson-host`: Jetson IP address
- `--jetson-port`: Jetson port
- `--ha-url`: Home Assistant URL
- `--ha-token`: Home Assistant token
- `--entities`: Path to ha_entities.json
- `--no-feedback`: Disable audio feedback
- `--debug`: Enable debug logging

### Running as a Service

#### Jetson (systemd)

Create `/etc/systemd/system/voice-assistant-server.service`:

```ini
[Unit]
Description=Voice Assistant Jetson Server
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/home/your_user/voice-assistant
Environment="PATH=/home/your_user/voice-assistant/venv/bin"
ExecStart=/home/your_user/voice-assistant/venv/bin/python run_jetson.py --entities ha_entities.json
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable voice-assistant-server
sudo systemctl start voice-assistant-server
```

#### Raspberry Pi (systemd)

Create `/etc/systemd/system/voice-assistant.service`:

```ini
[Unit]
Description=Voice Assistant Pi Client
After=network.target sound.target

[Service]
Type=simple
User=bernardoct
WorkingDirectory=/home/bernardoct/voice-assistant
Environment="PATH=/home/bernardoct/voice-assistant/venv/bin"
Environment="HA_TOKEN=your_token_here"
ExecStart=/home/bernardoct/voice-assistant/venv/bin/python run_pi.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable voice-assistant
sudo systemctl start voice-assistant
```

## Latency Optimization

The system is designed for minimal latency:

1. **Voice Activity Detection (VAD)**: Uses WebRTC VAD to detect speech boundaries quickly, avoiding unnecessary processing of silence.

2. **Streaming Audio**: Audio is streamed to the Jetson as it's captured, rather than waiting for the complete utterance.

3. **GPU Acceleration**: faster-whisper uses CTranslate2 for optimized inference on NVIDIA GPUs.

4. **Pattern Matching First**: Simple commands use fast regex matching; LLM is only used as fallback for complex commands.

5. **Persistent Connections**: WebSocket connections stay open to avoid connection overhead.

### Expected Latency

| Component | Latency |
|-----------|---------|
| VAD Detection | ~30ms |
| Audio Streaming | ~50ms |
| Whisper (base.en, GPU) | ~200-400ms |
| Intent Parsing | ~5ms |
| HA API Call | ~50-100ms |
| **Total** | **~350-600ms** |

## Troubleshooting

### No audio input detected
```bash
# Check available audio devices
python -c "import pyaudio; p = pyaudio.PyAudio(); [print(p.get_device_info_by_index(i)) for i in range(p.get_device_count())]"
```

### Connection refused to Jetson
```bash
# Check if server is running
curl -v ws://192.168.1.117:8765

# Check firewall
sudo ufw allow 8765
```

### CUDA out of memory
Use a smaller Whisper model:
```bash
python run_jetson.py --model tiny.en
```

## Project Structure

```
voice-assistant/
├── shared/
│   ├── __init__.py
│   └── protocol.py          # Message definitions
├── jetson/
│   ├── __init__.py
│   ├── config.py            # Jetson configuration
│   ├── server.py            # WebSocket server
│   ├── transcriber.py       # Whisper wrapper
│   └── intent_parser.py     # Intent parsing
├── pi/
│   ├── __init__.py
│   ├── config.py            # Pi configuration
│   ├── main.py              # Main application
│   ├── audio_capture.py     # Audio + VAD
│   ├── ha_controller.py     # Home Assistant API
│   ├── jetson_client.py     # WebSocket client
│   └── feedback.py          # Audio feedback
├── run_jetson.py            # Jetson entry point
├── run_pi.py                # Pi entry point
├── requirements-jetson.txt
├── requirements-pi.txt
└── README.md
```

## License

MIT License
