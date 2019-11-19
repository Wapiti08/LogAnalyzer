{
    "logs": [{
  "_index": "syslog-2019.06.01",
  "_type": "doc",
  "_id": "WXdFFWsBBm1tvzJxqnuE",
  "_version": 1,
  "_score": null,
  "_source": {
    "src_ip": "185.18.138.19",
    "src_port": "60425",
    "tags": [
      "syslog"
    ],
    "tls_profile": "TLS-Client-HTTPS.Standard",
    "msg_id": "2CFF-0000",
    "rcvd_bytes": "82401",
    "sent_bytes": "82401",
    "message": "<142>Jun  1 23:59:59 Member1 80D602937D0FA WG-HCC-EXT02 (2019-06-01T22:59:59) https-proxy[3056]: msg_id=\"2CFF-0000\" Allow 0-Updata 2-Optional-1 tcp 185.18.138.19 46.16.5.197 60425 443 msg=\"HTTPS Request\" proxy_act=\"HTTPS-Client.Standard\" tls_profile=\"TLS-Client-HTTPS.Standard\" tls_version=\"TLS_V12\" sni=\"\" cn=\"services.hertfordshire.gov.uk\" cert_issuer=\"CN=QuoVadis EV SSL ICA G3,O=QuoVadis Limited,C=BM\" cert_subject=\"CN=services.hertfordshire.gov.uk,O=Hertfordshire County Council,L=HERTFORD,ST=Hertfordshire,C=GB,serialNumber=Government Entity,businessCategory=Government Entity,jurisdictionC=GB\" action=\"allow\" app_id=\"0\" app_cat_id=\"0\" sent_bytes=\"82401\" rcvd_bytes=\"82401\"  geo_src=\"GBR\"  (HTTPS-Services-00)",
    "proxy_action": [
      "HTTPS-Client.Standard",
      "allow"
    ],
    "dest_ip": "46.16.5.197",
    "cert_subject": "CN=services.hertfordshire.gov.uk,O=Hertfordshire County Council,L=HERTFORD,ST=Hertfordshire,C=GB,serialNumber=Government Entity,businessCategory=Government Entity,jurisdictionC=GB",
    "cn": "services.hertfordshire.gov.uk",
    "syslog_pri": "142",
    "type": "syslog",
    "@version": "1",
    "@timestamp": "2019-06-01T22:59:59.989Z",
    "host": "192.168.12.65",
    "protocol": "tcp",
    "dest_port": "443",
    "tls_version": "TLS_V12",
    "syslog_timestamp": "Jun  1 23:59:59",
    "cert_issuer": "CN=QuoVadis EV SSL ICA G3,O=QuoVadis Limited,C=BM",
    "proxy_message": [
      "Allow",
      "0-Updata",
      "2-Optional-1",
      "HTTPS Request"
    ],
    "syslog_hostname": "WG-HCC-EXT02",
    "app_id": "0"
  },
  "fields": {
    "@timestamp": [
      "2019-06-01T22:59:59.989Z"
    ]
  },
  "sort": [
    1559429999989
  ]
},
{
  "_index": "syslog-2019.06.01",
  "_type": "doc",
  "_id": "W3dFFWsBBm1tvzJxqnuE",
  "_version": 1,
  "_score": null,
  "_source": {
    "@version": "1",
    "@timestamp": "2019-06-01T23:00:00.000Z",
    "src_ip": "206.124.114.91",
    "host": "192.168.12.65",
    "protocol": "tcp",
    "tags": [
      "syslog"
    ],
    "duration": "1860",
    "msg_id": "3000-0151",
    "rcvd_bytes": "15086",
    "firewall_name": "HTTPS-00",
    "syslog_timestamp": "Jun  1 23:59:59",
    "sent_bytes": "11656",
    "time": "2019-06-01T22:59:59",
    "message": "<142>Jun  1 23:59:59 Member1 80D602937D0FA WG-HCC-EXT02 (2019-06-01T22:59:59) firewall: msg_id=\"3000-0151\" Allow 2-Optional-1 0-Updata tcp 192.168.100.248 206.124.114.91 37738 443  geo_dst=\"CAN\" duration=\"1860\" sent_bytes=\"11656\" rcvd_bytes=\"15086\"  (HTTPS-00)",
    "geo_dst": "CAN",
    "dest_ip": "192.168.100.248",
    "syslog_pri": "142",
    "type": "syslog",
    "syslog_hostname": "Member1",
    "firewall_response": [
      "Allow",
      "2-Optional-1",
      "0-Updata",
      "37738",
      "443"
    ]
  },
  "fields": {
    "@timestamp": [
      "2019-06-01T23:00:00.000Z"
    ],
    "time": [
      "2019-06-01T22:59:59.000Z"
    ]
  },
  "highlight": {
    "firewall_name": [
      "@kibana-highlighted-field@HTTPS@/kibana-highlighted-field@-00"
    ],
    "message": [
      "<142>Jun  1 23:59:59 Member1 80D602937D0FA WG-HCC-EXT02 (2019-06-01T22:59:59) firewall: msg_id=\"3000-0151\" Allow 2-Optional-1 0-Updata tcp 192.168.100.248 206.124.114.91 37738 443  geo_dst=\"CAN\" duration=\"1860\" sent_bytes=\"11656\" rcvd_bytes=\"15086\"  (@kibana-highlighted-field@HTTPS@/kibana-highlighted-field@-00)"
    ]
  },
  "sort": [
    1559430000000
  ]
},
{
  "_index": "syslog-2019.06.01",
  "_type": "doc",
  "_id": "WHdFFWsBBm1tvzJxqnuE",
  "_version": 1,
  "_score": null,
  "_source": {
    "@version": "1",
    "@timestamp": "2019-06-01T22:59:59.985Z",
    "src_ip": "10.164.11.2",
    "src_port": "31554",
    "host": "192.168.12.65",
    "protocol": "tcp",
    "tags": [
      "syslog"
    ],
    "dest_port": "443",
    "msg_id": "2DFF-0004",
    "syslog_timestamp": "Jun  1 23:59:59",
    "message": "<142>Jun  1 23:59:59 Member1 80D602937D0FA WG-HCC-EXT02 (2019-06-01T22:59:59) tcp-udp-proxy[3060]: msg_id=\"2DFF-0004\" Allow 1-Trusted 0-Updata tcp 10.164.11.2 206.124.114.91 31554 443 msg=\"ProxyReplace: IP protocol\" proxy_act=\"TCP-UDP-Proxy.1\" rule_name=\"HTTPS-Client.4\" new_action=\"HTTPS-Client.4\"  geo_dst=\"CAN\"  (GoodTechnology-00)",
    "proxy_action": "TCP-UDP-Proxy.1",
    "dest_ip": "206.124.114.91",
    "proxy_message": [
      "Allow",
      "1-Trusted",
      "0-Updata",
      "ProxyReplace: IP protocol"
    ],
    "syslog_pri": "142",
    "type": "syslog",
    "syslog_hostname": "WG-HCC-EXT02"
  },
  "fields": {
    "@timestamp": [
      "2019-06-01T22:59:59.985Z"
    ]
  },
  "highlight": {
    "message": [
      "<142>Jun  1 23:59:59 Member1 80D602937D0FA WG-HCC-EXT02 (2019-06-01T22:59:59) tcp-udp-@kibana-highlighted-field@proxy@/kibana-highlighted-field@[3060]: msg_id=\"2DFF-0004\" Allow 1-Trusted 0-Updata tcp 10.164.11.2 206.124.114.91 31554 443 msg=\"ProxyReplace: IP protocol\" proxy_act=\"TCP-UDP-@kibana-highlighted-field@Proxy@/kibana-highlighted-field@.1\" rule_name=\"@kibana-highlighted-field@HTTPS@/kibana-highlighted-field@-Client.4\" new_action=\"@kibana-highlighted-field@HTTPS@/kibana-highlighted-field@-Client.4\"  geo_dst=\"CAN\"  (GoodTechnology-00)"
    ],
    "proxy_action": [
      "TCP-UDP-@kibana-highlighted-field@Proxy@/kibana-highlighted-field@.1"
    ]
  },
  "sort": [
    1559429999985
  ]
},
{
  "_index": "syslog-2019.06.01",
  "_type": "doc",
  "_id": "V3dFFWsBBm1tvzJxqnuE",
  "_version": 1,
  "_score": null,
  "_source": {
    "@version": "1",
    "@timestamp": "2019-06-01T22:59:59.984Z",
    "src_ip": "10.164.11.2",
    "src_port": "31553",
    "host": "192.168.12.65",
    "protocol": "tcp",
    "tags": [
      "syslog"
    ],
    "dest_port": "443",
    "msg_id": "2DFF-0004",
    "syslog_timestamp": "Jun  1 23:59:59",
    "message": "<142>Jun  1 23:59:59 Member1 80D602937D0FA WG-HCC-EXT02 (2019-06-01T22:59:59) tcp-udp-proxy[3061]: msg_id=\"2DFF-0004\" Allow 1-Trusted 0-Updata tcp 10.164.11.2 206.124.114.91 31553 443 msg=\"ProxyReplace: IP protocol\" proxy_act=\"TCP-UDP-Proxy.1\" rule_name=\"HTTPS-Client.4\" new_action=\"HTTPS-Client.4\"  geo_dst=\"CAN\"  (GoodTechnology-00)",
    "proxy_action": "TCP-UDP-Proxy.1",
    "dest_ip": "206.124.114.91",
    "proxy_message": [
      "Allow",
      "1-Trusted",
      "0-Updata",
      "ProxyReplace: IP protocol"
    ],
    "syslog_pri": "142",
    "type": "syslog",
    "syslog_hostname": "WG-HCC-EXT02"
  },
  "fields": {
    "@timestamp": [
      "2019-06-01T22:59:59.984Z"
    ]
  },
  "highlight": {
    "message": [
      "<142>Jun  1 23:59:59 Member1 80D602937D0FA WG-HCC-EXT02 (2019-06-01T22:59:59) tcp-udp-@kibana-highlighted-field@proxy@/kibana-highlighted-field@[3061]: msg_id=\"2DFF-0004\" Allow 1-Trusted 0-Updata tcp 10.164.11.2 206.124.114.91 31553 443 msg=\"ProxyReplace: IP protocol\" proxy_act=\"TCP-UDP-@kibana-highlighted-field@Proxy@/kibana-highlighted-field@.1\" rule_name=\"@kibana-highlighted-field@HTTPS@/kibana-highlighted-field@-Client.4\" new_action=\"@kibana-highlighted-field@HTTPS@/kibana-highlighted-field@-Client.4\"  geo_dst=\"CAN\"  (GoodTechnology-00)"
    ],
    "proxy_action": [
      "TCP-UDP-@kibana-highlighted-field@Proxy@/kibana-highlighted-field@.1"
    ]
  },
  "sort": [
    1559429999984
  ]
},
{
  "_index": "syslog-2019.06.01",
  "_type": "doc",
  "_id": "VXdFFWsBBm1tvzJxqnuE",
  "_version": 1,
  "_score": null,
  "_source": {
    "@version": "1",
    "@timestamp": "2019-06-01T22:59:59.975Z",
    "src_ip": "46.16.1.228",
    "host": "192.168.12.65",
    "protocol": "tcp",
    "tags": [
      "syslog"
    ],
    "msg_id": "3000-0148",
    "syslog_timestamp": "Jun  1 22:59:59",
    "time": "2019-06-01T22:59:59",
    "message": "<140>Jun  1 22:59:59 Member2 80D60293B373B WG-HCC-A1EXT01 (2019-06-01T22:59:59) firewall: msg_id=\"3000-0148\" Allow 0-Updata 2-Optional-1 60 tcp 20 50 185.18.138.3 46.16.1.228 54597 443 offset 10 S 2992250541 win 4210  geo_src=\"GBR\"  (HTTPS-Services-00)",
    "dest_ip": "185.18.138.3",
    "syslog_pri": "140",
    "type": "syslog",
    "syslog_hostname": "Member2",
    "firewall_response": [
      "Allow",
      "0-Updata",
      "2-Optional-1",
      "60",
      "20",
      "50",
      "54597",
      "443"
    ]
  },
  "fields": {
    "@timestamp": [
      "2019-06-01T22:59:59.975Z"
    ],
    "time": [
      "2019-06-01T22:59:59.000Z"
    ]
  },
  "highlight": {
    "message": [
      "<140>Jun  1 22:59:59 Member2 80D60293B373B WG-HCC-A1EXT01 (2019-06-01T22:59:59) firewall: msg_id=\"3000-0148\" Allow 0-Updata 2-Optional-1 60 tcp 20 50 185.18.138.3 46.16.1.228 54597 443 offset 10 S 2992250541 win 4210  geo_src=\"GBR\"  (@kibana-highlighted-field@HTTPS@/kibana-highlighted-field@-Services-00)"
    ]
  },
  "sort": [
    1559429999975
  ]
},
{
  "_index": "syslog-2019.06.01",
  "_type": "doc",
  "_id": "S3dFFWsBBm1tvzJxqnuE",
  "_version": 1,
  "_score": null,
  "_source": {
    "@version": "1",
    "@timestamp": "2019-06-01T22:59:59.953Z",
    "src_ip": "46.16.5.197",
    "host": "192.168.12.65",
    "protocol": "tcp",
    "tags": [
      "syslog"
    ],
    "msg_id": "3000-0148",
    "syslog_timestamp": "Jun  1 23:59:59",
    "time": "2019-06-01T22:59:59",
    "message": "<140>Jun  1 23:59:59 Member1 80D602937D0FA WG-HCC-EXT02 (2019-06-01T22:59:59) firewall: msg_id=\"3000-0148\" Allow 0-Updata 2-Optional-1 60 tcp 20 53 185.18.138.19 46.16.5.197 60448 443 offset 10 S 3852705778 win 4210  geo_src=\"GBR\"  (HTTPS-Services-00)",
    "dest_ip": "185.18.138.19",
    "syslog_pri": "140",
    "type": "syslog",
    "syslog_hostname": "Member1",
    "firewall_response": [
      "Allow",
      "0-Updata",
      "2-Optional-1",
      "60",
      "20",
      "53",
      "60448",
      "443"
    ]
  },
  "fields": {
    "@timestamp": [
      "2019-06-01T22:59:59.953Z"
    ],
    "time": [
      "2019-06-01T22:59:59.000Z"
    ]
  },
  "highlight": {
    "message": [
      "<140>Jun  1 23:59:59 Member1 80D602937D0FA WG-HCC-EXT02 (2019-06-01T22:59:59) firewall: msg_id=\"3000-0148\" Allow 0-Updata 2-Optional-1 60 tcp 20 53 185.18.138.19 46.16.5.197 60448 443 offset 10 S 3852705778 win 4210  geo_src=\"GBR\"  (@kibana-highlighted-field@HTTPS@/kibana-highlighted-field@-Services-00)"
    ]
  },
  "sort": [
    1559429999953
  ]
},
{
  "_index": "syslog-2019.06.01",
  "_type": "doc",
  "_id": "OHdFFWsBBm1tvzJxqnuE",
  "_version": 1,
  "_score": null,
  "_source": {
    "@version": "1",
    "@timestamp": "2019-06-01T22:59:59.947Z",
    "src_ip": "46.16.1.228",
    "host": "192.168.12.65",
    "protocol": "tcp",
    "tags": [
      "syslog"
    ],
    "msg_id": "3000-0148",
    "syslog_timestamp": "Jun  1 22:59:59",
    "time": "2019-06-01T22:59:59",
    "message": "<140>Jun  1 22:59:59 Member2 80D60293B373B WG-HCC-A1EXT01 (2019-06-01T22:59:59) firewall: msg_id=\"3000-0148\" Allow 0-Updata 2-Optional-1 60 tcp 20 50 185.18.138.3 46.16.1.228 52517 443 offset 10 S 2758198178 win 4210  geo_src=\"GBR\"  (HTTPS-Services-00)",
    "dest_ip": "185.18.138.3",
    "syslog_pri": "140",
    "type": "syslog",
    "syslog_hostname": "Member2",
    "firewall_response": [
      "Allow",
      "0-Updata",
      "2-Optional-1",
      "60",
      "20",
      "50",
      "52517",
      "443"
    ]
  },
  "fields": {
    "@timestamp": [
      "2019-06-01T22:59:59.947Z"
    ],
    "time": [
      "2019-06-01T22:59:59.000Z"
    ]
  },
  "highlight": {
    "message": [
      "<140>Jun  1 22:59:59 Member2 80D60293B373B WG-HCC-A1EXT01 (2019-06-01T22:59:59) firewall: msg_id=\"3000-0148\" Allow 0-Updata 2-Optional-1 60 tcp 20 50 185.18.138.3 46.16.1.228 52517 443 offset 10 S 2758198178 win 4210  geo_src=\"GBR\"  (@kibana-highlighted-field@HTTPS@/kibana-highlighted-field@-Services-00)"
    ]
  },
  "sort": [
    1559429999947
  ]
}]
}
