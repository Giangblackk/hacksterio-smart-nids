from collections import Counter
from scapy.all import sniff


packet_counter = Counter()


def custom_action(packet):
    key = tuple(sorted([packet[0][1].src, packet[0][1].dst],))
    packet_counter.update([key])
    return f"Packet #{sum(packet_counter.values())}: {packet[0][1].src} ==> {packet[0][1].dst}"


sniff(filter="ip", prn=custom_action, count=10, iface="eno1")

print(
    "\n".join(
        f"{f'{key[0]} <--> {key[1]}'}: {count}" for key, count in packet_counter.items()
    )
)
