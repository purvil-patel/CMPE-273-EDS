# Real-time Communication Using WebSockets

This project demonstrates the use of WebSockets for real-time, full-duplex communication between a client and a server. WebSockets provide a more efficient way for persistent communication compared to traditional HTTP.


https://github.com/purvil-patel/CMPE-273-EDS/assets/67355397/8ad419d7-cce6-4866-b370-c39620dc2919


**Prerequisites**

Python 3.x
web sockets library. Install it using: ```pip install websockets```

**Running the Code**

Start the WebSocket Server:

Run the socketserver.py: ```python socketserver.py```

This starts the WebSocket server on localhost at port 8765. The server listens for incoming messages from the client, appends a random number to each message, and sends it back.

Run the WebSocket Client:

In a separate terminal or command prompt, run the socketclient.py: ```python socketclient.py```

The client will make 10,000 requests to the server, demonstrating the efficiency and reliability of WebSockets for real-time communication.

