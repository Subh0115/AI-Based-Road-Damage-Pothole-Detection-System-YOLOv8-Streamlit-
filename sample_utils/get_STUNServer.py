import random


def getSTUNServer() -> str:
	"""Return a public STUN server hostname.

	A small rotating list to reduce the chance of a single provider outage.
	"""
	stun_hosts = [
		"stun.l.google.com:19302",
		"stun1.l.google.com:19302",
		"stun2.l.google.com:19302",
		"stun3.l.google.com:19302",
		"stun4.l.google.com:19302",
	]
	return random.choice(stun_hosts)


