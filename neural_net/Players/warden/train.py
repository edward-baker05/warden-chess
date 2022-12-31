from warden_engine import WardenEngine, compare

def main():
    warden = WardenEngine()
    try:
        warden.load(r"C:\Users\ed9ba\Documents\Coding\NEA\Warden\neural_net\Players\warden\warden_weights.h5")
    except FileNotFoundError:
        pass
    warden.train()
    warden.save(r"C:\Users\ed9ba\Documents\Coding\NEA\Warden\neural_net\Players\warden\warden_weights_new.h5")
    
    try:
        compare()
    except FileNotFoundError:
        print("Could not find weights file")
        pass

if __name__ == '__main__':
    main()
