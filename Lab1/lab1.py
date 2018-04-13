import cv2

def zad1():
    cap = cv2.VideoCapture(0)
    key = ord('a')
    while key != ord('q'):
        ret, frame = cap.read()
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_filtered = cv2.GaussianBlur(img_gray, (7, 7), 1.5)
        img_edges = cv2.Canny(img_filtered, 0, 30, 3)
        cv2.imshow('result', img_edges)
        key = cv2.waitKey(30)
    cap.release()
    cv2.destroyAllWindows()


def zad2():
    img_c = cv2.imread('flaming.jpeg', cv2.IMREAD_COLOR)
    img_g = cv2.imread('flaming.jpeg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('flaming color', img_c)
    cv2.imshow('flaming gray', img_g)
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()
    elif key == ord('s'):
        cv2.imwrite('flaming_gray.jpg', img_g)
        cv2.destroyAllWindows()


def zad3_a():
    img_c = cv2.imread('flaming.jpeg', cv2.IMREAD_COLOR)
    img_g = cv2.imread('flaming.jpeg', cv2.IMREAD_GRAYSCALE)
    pixel_c = img_c[220,270]
    pixel_g = img_g[220,270]
    print('Pixel color', pixel_c)
    print('Shape color img ', img_c.shape)
    print('Pixel grey', pixel_g)
    print('Shape grey img ', img_g.shape)


def zad3_b():
    img = cv2.imread('img.png', cv2.IMREAD_COLOR)
    b, g, r = cv2.split(img)
    key = ord('a')
    while key != ord('q'):
        cv2.imshow('blue', b)
        cv2.imshow('red', r)
        cv2.imshow('green', g)
        key = cv2.waitKey(30)
    cv2.destroyAllWindows()


def zad4():
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))
    while(cap.isOpened()):
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def zad4_b():
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))
    while(cap.isOpened()):
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            out.write(frame)
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def zad5():
    img_c = cv2.imread('flaming.jpeg', cv2.IMREAD_COLOR)
    img = cv2.rectangle(img_c, (200,180),(320,350),(0,200,0),10)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'Umiem obrazy!', (80,800), font, 2, (0,255,255), 10, cv2.LINE_AA )
    cv2.imshow('flaming gray', img)
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()


def zad6():
    cap = cv2.VideoCapture('Wildlife.mp4')
    while (cap.isOpened()):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        while cv2.waitKey(1) & 0xFF != ord(' '):
            pass
    cap.release()
    cv2.destroyAllWindows()

def zad7():
    i = 0
    while 0xFF != ord('q'):
        name = str(i) + '.jpg'
        print(name)
        img = cv2.imread(name, cv2.IMREAD_COLOR)
        cv2.imshow('1', img)
        if  cv2.waitKey(1) & 0xFF == ord('n'):
            if i < 3 :
                cv2.destroyAllWindows()
                i= i+1
                print(i)
            else:
                cv2.destroyAllWindows()
                i = 0
    cv2.destroyAllWindows()



if __name__ == '__main__':
    zad1()

