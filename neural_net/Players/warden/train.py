from warden_engine import WardenEngine

def main():
    warden = WardenEngine()
    try:
        warden.load(r"C:\Users\ed9ba\Documents\Coding\NEA\Warden\neural_net\Players\warden\warden_weights.h5")
    except FileNotFoundError:
        pass
    warden.train()
    warden.save(r"C:\Users\ed9ba\Documents\Coding\NEA\Warden\neural_net\Players\warden\warden_weights.h5")

if __name__ == '__main__':
    main()