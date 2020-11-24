from collections import Counter
from scapy.all import sniff, IP, IPv6, TCP, UDP, ARP, ICMP
# from scapy.layers.inet import IP
# from scapy.layers.inet import IPv6


packet_counter = Counter()


def custom_action(packet):
    key = tuple(sorted([packet[0][1].src, packet[0][1].dst],))
    packet_counter.update([key])
    return f"Packet #{sum(packet_counter.values())}: {packet[0][1].src} ==> {packet[0][1].dst}"

def custom_packet_parser(packet):
    IPtype = None # undefined IPtype
    timestamp = packet.time
    framelen = len(packet)

    if packet.haslayer(IP):
        srcIP = packet[IP].src
        dstIP = packet[IP].dst
        IPtype = 0
    elif packet.haslayer(IPv6):
        srcIP = packet[IPv6].src
        dstIP = packet[IPv6].dst
        IPtype = 1
    else:
        srcIP = ""
        dstIP = ""
    
    if packet.haslayer(TCP):
        srcproto = str(packet[TCP].sport)
        dstproto = str(packet[TCP].dport)
    elif packet.haslayer(UDP):
        srcproto = str(packet[UDP].sport)
        dstproto = str(packet[UDP].dport)
    else:
        srcproto = ""
        dstproto = ""
    
    srcMAC = packet.src
    dstMAC = packet.dst
    
    if srcproto == "" and dstproto == "": #not tcp or udp -> L2/L1 protocol
        if packet.haslayer(ARP):
            srcproto = "arp"
            dstproto = "arp"
            srcIP = packet[ARP].psrc
            dstIP = packet[ARP].pdst
            IPtype = 0
        elif packet.haslayer(ICMP):
            srcproto = "icmp"
            dstproto = "icmp"
            IPType = 1
        elif srcIP + srcproto + dstIP + dstproto == "": # other protocol
            srcIP = packet.src # get src MAC
            dstIP = packet.dst # get dst MAC
    return f"{IPtype},{srcMAC},{dstMAC},{srcIP},{srcproto},{dstIP},{dstproto},{int(framelen)},{float(timestamp)}"


# sniff(filter="ip", prn=custom_action, count=10, iface="eno1")
sniff(filter="ip", prn=custom_packet_parser, count=10, iface="eno1")

print(
    "\n".join(
        f"{f'{key[0]} <--> {key[1]}'}: {count}" for key, count in packet_counter.items()
    )
)
