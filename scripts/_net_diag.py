"""Read-only: why can't WSL reach api.anthropic.com? Tests raw IPv4 internet,
DNS resolution, and IPv4/IPv6 connects. Safe to delete."""
import socket


def try_connect(label, sa, fam):
    try:
        s = socket.socket(fam, socket.SOCK_STREAM)
        s.settimeout(6)
        s.connect(sa)
        print(f"  {label} OK -> {sa[0]}:{sa[1]}")
        s.close()
    except Exception as e:
        print(f"  {label} FAILED: {e!r}")


print("=== raw IPv4 internet (no DNS) ===")
try_connect("8.8.8.8:443", ("8.8.8.8", 443), socket.AF_INET)
try_connect("1.1.1.1:443", ("1.1.1.1", 443), socket.AF_INET)

HOST, PORT = "api.anthropic.com", 443
print("=== getaddrinfo api.anthropic.com ===")
try:
    for fam, _, _, _, sa in socket.getaddrinfo(HOST, PORT, proto=socket.IPPROTO_TCP):
        print(f"  {'IPv6' if fam == socket.AF_INET6 else 'IPv4'} {sa[0]}")
except Exception as e:
    print("  getaddrinfo FAILED:", repr(e))

for fam, label in ((socket.AF_INET, "IPv4"), (socket.AF_INET6, "IPv6")):
    print(f"=== {label} connect api.anthropic.com ===")
    try:
        sa = socket.getaddrinfo(HOST, PORT, fam, socket.SOCK_STREAM)[0][4]
        try_connect(label, sa, fam)
    except Exception as e:
        print(f"  {label} resolve FAILED: {e!r}")
