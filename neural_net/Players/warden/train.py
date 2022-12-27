from warden_engine import WardenEngine

if __name__ == '__main__':
    warden = WardenEngine()
    warden.load(r"C:\Users\ed9ba\Documents\Coding\NEA\Warden\neural_net\Players\warden\warden_weights.h5")
    warden.train()
    warden.save(r"C:\Users\ed9ba\Documents\Coding\NEA\Warden\neural_net\Players\warden\warden_weights.h5")