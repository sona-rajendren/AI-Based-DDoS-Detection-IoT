Data source: **edge-iiotset**
Total rows used: 100000 (train 80000, test 20000)
Features (10): tcp_syn, tcp_ack, tcp_rst, tcp_fin, tcp_len, udp_present, icmp_present, http_present, mqtt_msgtype, mqtt_len

| Class | Samples |
|---|---|
| Normal | 50000 |
| DDoS_TCP | 12500 |
| DDoS_UDP | 12500 |
| DDoS_ICMP | 12500 |
| DDoS_HTTP | 12500 |