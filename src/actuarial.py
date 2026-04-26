def compute_prime_pure(generated, trigger, indemnite):
    prob = float((generated > trigger).mean())
    prime = prob * indemnite
    return prob, prime


def print_actuarial_report(generated, trigger, indemnite, label='Modèle'):
    prob, prime = compute_prime_pure(generated, trigger, indemnite)
    print(f"--- ANALYSE ASSURANCE ({label}) ---")
    print(f"Seuil paramétrique            : {trigger}")
    print(f"Probabilité d'événement       : {prob * 100:.2f}%")
    print(f"Prime Pure annuelle suggérée  : {prime:.2f} €")
    return prob, prime
