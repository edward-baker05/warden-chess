from warden_engine import WardenEngine

def main():
    warden = WardenEngine()
    warden.load(r"C:\Users\ed9ba\Documents\Coding\NEA\Warden\neural_net\Players\warden\warden_weights.h5")
    warden.train()
    warden.save(r"C:\Users\ed9ba\Documents\Coding\NEA\Warden\neural_net\Players\warden\warden_weights.h5")

if __name__ == '__main__':
    main()