import socket
import time
import threading

from touchscreen import TouchScreen

def receive_data(conn, addr):
    print(f'Connected by {addr}')
    while True:
        data = conn.recv(1024)
        if data:
            print(data.decode())

def send_data(conn, addr):
    while True:
        position = touchscreen.position()
        message = f"{position[0]} {position[1]}"
        conn.sendall(message.encode())
        time.sleep(1)  # Send a message every second

def handle_client(conn, addr):
    print(f'Connected by {addr}')
    while True:
        data = conn.recv(1024)
        if data:
            data = data.decode().split()
            print(data)
            if data[0] == "0":
                W,H = int(data[1]),int(data[2])

                message = "0" # Starting calibration
                conn.sendto(message.encode(), addr)
                time.sleep(2)

                touchscreen.calibrate_point(W, H, "tl")
                message = "1" # Top-Left calibration complete
                conn.sendto(message.encode(), addr)
                time.sleep(2)
                
                touchscreen.calibrate_point(W, H, "br")
                message = "2" # Bottom-right calibration complete
                touchscreen.recalibrate()
                conn.sendto(message.encode(), addr)
                

            elif data[0] == "1":
                print("getting position")
                position = touchscreen.pixels()
                if position == -1:
                    message = f"{position}"
                else:
                    message = f"{position[0]} {position[1]}"
                conn.sendto(message.encode(), addr)

def send_position(conn, addr):
    while True:
        print("getting position")
        position = touchscreen.pixels()
        if position == -1:
            message = f"{position}"
        else:
            message = f"{position[0]} {position[1]}"
        conn.sendto(message.encode(), addr)
        time.sleep(0.5)

def start_server(host='0.0.0.0', port=123):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((host, port))
        server_socket.listen()
        print(f'Server listening on {host}:{port}')
        
        conn, addr = server_socket.accept()
        
        position_thread = threading.Thread(target=send_position, args=(conn, addr,))
        handler_thread = threading.Thread(target=handle_client, args=(conn, addr,))
        
        position_thread.start()
        handler_thread.start()
        
        handler_thread.join()
        position_thread.join()
        

touchscreen = TouchScreen()
if __name__ == "__main__":
    try:
        start_server()
    except KeyboardInterrupt:
        touchscreen.cleanup()

